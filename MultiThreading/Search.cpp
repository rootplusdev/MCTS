#include "Search.h"
#include<random>
#include<fstream>
#include<iomanip>
#include <time.h>
#include<pybind11/pybind11.h>
namespace py = pybind11;

// 构造函数初始化
Search::Search(float cPuct, unsigned ms)
{

	deviceType = at::kCPU;
	if (torch::cuda::is_available())
		deviceType = at::kCUDA;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available!" << std::endl;
	}
	this->cPuct = cPuct;
	this->SearchTime = ms;
	this->model = torch::jit::load("model/20b128c_renju.pt", deviceType);
	//this->model.to(torch::kHalf);
	std::cout << "Model Load Successfully!\n";
	this->model.eval();
	this->model.forward({ torch::zeros({ 1, 3, 15, 15 }).to(torch::kCUDA) });
	this->root = new Node(1.0, this->cPuct, nullptr);
	this->nnserver = std::thread(&Search::nn_evaluate, this);  //NN线程
}

// 析构函数
Search::~Search() 
{
	this->finished++;
	this->asySearchForward.notify_all();
	this->nnserver.join();
	//this->finished--;

	delete root;
}

void Search::getRank(Board* board) {
	auto feature = board->getFeature();
	//feature.unsqueeze_(0);
	std::cout << feature.sizes() << std::endl;
	torch::IValue policyValue = model.forward({ feature.to(deviceType) });
	auto tuplePolicyAndValue = policyValue.toTuple();
	auto policy = tuplePolicyAndValue->elements()[0].toTensor().data().to(at::kCPU);
	policy = policy.exp();
	policy = policy[0];
	policy = policy.argsort();
	std::ofstream outfile;
	outfile.open("fivePointRank.txt", std::ios::out);
	for (size_t i = 0; i < 225; i++)
	{
		outfile <<std::fixed << std::setprecision(225) << policy[i].item().toInt() << " ";
	}
	outfile.close();
}


void Search::calculation(const Board* board) {
	srand((unsigned)time(0));
	//float start = clock();
	//float copyTime = 0.0;
	// for (size_t i = 0; i < nIter; i++)
	while (this->is_end.load() == 0)
	{
		//float t1 = clock();
		std::unique_ptr<Board> copiedBoard{ new Board(board) };

		//copyTime += (clock() - t1);
		Node* node = root;

		//	_InternalNodeLoop:
		while (!node->isLeafNode())
		{
			std::pair<int, Node*> actionAndNode = node->select();

			// 判断是否动作可行
			bool flag = false;
			for (auto each : copiedBoard->availableAction)
			{
				if (actionAndNode.first == each)
				{
					flag = true;
				}
			}
			if (!flag)
			{
				std::cout << "动作不可选" << std::endl;
				// assert("动作不可选");
			}
			copiedBoard->doAction(actionAndNode.first);
			node = actionAndNode.second;
		}
		int isOverAndWinner = copiedBoard->isGameOver();
		float value = 0.0f;
		if (isOverAndWinner == board->notOver)
		{
			std::unique_lock<std::shared_mutex> lk(node->nodeLock);

			if (node->isLeafNode(false))
			{
				auto policyValue = predict_commit(copiedBoard);
				value = policyValue.second;
				auto policy = policyValue.first;
				// policy = policy.exp();
				auto actionVector = copiedBoard->availableAction;
				std::vector<float> p;
				for (auto a : actionVector)
				{
					p.push_back(policy[a].item().toFloat());
				}
				if (actionVector.size() != p.size())
				{
					assert("Action and P Length Is Not Equal!");
				}
				std::map<int, float> actionProb;
				for (size_t i = 0; i < actionVector.size(); i++)
				{
					actionProb.insert(std::pair<int, float>(actionVector[i], p[i]));
				}

				node->expand(actionProb);
			}
			else
				continue; //goto _InternalNodeLoop;
		}
		else
		{
			if (copiedBoard->currentPlayer == isOverAndWinner)
			{
				value = 1.0f;
			}
			else if (isOverAndWinner == copiedBoard->EMPTY || isOverAndWinner == board->notOver) {
				value = 0.0f;
			}
			else
			{
				value = -1.0f;
			}
		}
		node->backup(value);
	}
}


int Search::getAction(const Board* board)
{
	py::gil_scoped_release release;	// 释放全局锁
	resetRoot();

	srand((unsigned)time(0));
	clock_t start = clock();
	//torch::IValue policyValue1 = predict(board);
	//auto tuplePolicyAndValue1 = policyValue1.toTuple();
	//auto policy1 = tuplePolicyAndValue1->elements()[0].toTensor();
	//std::cout << "策略最佳：" << policy1.argmax() << std::endl;

	std::vector<std::thread> mutiThread;
	for (size_t i = 0; i < numThread; i++)
	{
		mutiThread.push_back(std::thread(&Search::calculation, this, board));
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(this->SearchTime));
	//Sleep(this->SearchTime);
	/*if (this->root->Q > -0.95) {
		Sleep((this->search_time - 6) * 1000);
	}*/
	this->is_end++;
	//等待搜索线程结束
	for (size_t i = 0; i < numThread; i++)
	{
		mutiThread[i].join();
	}
	this->is_end--;
	//搜索结束



	float T = 0.5f;
	if (board->numPieces > 30)
	{
		T = 0.001f;
	}
	std::vector<float> visits;
	std::vector<int> actions;
	for (std::map<int, Node*>::iterator iter = root->children.begin();
		iter != root->children.end(); iter++)
	{
		visits.push_back(static_cast<float>(iter->second->N));
		actions.push_back(iter->first);
	}

	int action = -1;
	if (!visits.empty())
	{
		auto pi = getPi(visits, T);
		action = unequalProbSample(actions, pi);

		std::cout << "[INFO TIME ]: " << clock() - start << std::endl;
		std::cout << "[INFO R.COUNT]: " << root->N << std::endl;
		std::cout << "[INFO C.COUNT]: " << root->children[action]->N << std::endl;
		std::cout << "[INFO NPS]: " << root->N * 1000 / (clock() - start) << std::endl;
		std::cout << "[INFO ACTIONS]: " << board->availableAction.size() << std::endl;
		std::cout << "[INFO PIECES]: " << board->numPieces << std::endl;

		std::cout << "[INFO R.WinRate]: " << (root->Q + 1.0f) * 50 << std::endl;
		std::cout << "[INFO C.WinRate]: " << (-root->children[action]->Q + 1.0f) * 50 << std::endl;
	}
	nodeCount = root->N;
	winRate = (-root->children[action]->Q + 1.0f) * 50;
	py::gil_scoped_acquire acquire;	// 重新添加全局锁
	return action;
}

std::vector<float> Search::getPi(std::vector<float>& visits, float T)
{
	auto opts = torch::TensorOptions().dtype(torch::kFloat);
	auto visitTensor = torch::from_blob(visits.data(),
		{ int(visits.size()) }, opts);
	auto x = (1 / T) * (visitTensor + 1E-10).log();
	x = x - x.max();
	x = x.exp();
	auto pi = x / x.sum();
	// std::cout << pi << std::endl;
	std::vector<float> p(pi.data_ptr<float>(),
		pi.data_ptr<float>() + pi.numel());
	// std::cout << p << std::endl;
	return p;
}

void Search::resetRoot()
{
	delete this->root;
	this->root = new Node(1.0, 1.0, nullptr);
}

// 以向量的内容作为下标切片另一个向量
// 仅仅是遍历，想不到什么好办法
std::vector<float> Search::getElementsByVector(
	std::vector<float> mainVector,
	std::vector<int> indexVector
)
{
	std::vector<float> targrtVector;
	if (mainVector.size() != indexVector.size())
	{
		assert("[INFO: ERROR] Vector Length Is Not Equal!");
	}
	for (size_t i = 0; i < indexVector.size(); i++)
	{
		targrtVector.push_back(mainVector[indexVector[i]]);
	}
	return targrtVector;
}

//按照向量的概率分布选主向量的数
int Search::unequalProbSample(
	std::vector<int> mainVector,
	std::vector<float> probVector
)
{
	if (mainVector.size() != probVector.size())
	{
		assert("[INFO: ERROR] Vector Length Is Not Equal!");
	}
	//float max = -1000;
	//int index = 0;
	//for (size_t i = 0; i < probVector.size(); i++)
	//{
	//	if (max < probVector[i])
	//	{
	//		max = probVector[i];
	//		index = i;
	//	}
	//}
	//std::cout << "Action:" << mainVector[index] << std::endl;
	std::discrete_distribution<int> dis(probVector.begin(), probVector.end());
	long long numRandom = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::default_random_engine rng(static_cast<unsigned>(numRandom & 0xffffffffull));
	int bestIndex = dis(rng);
	// std::cout << bestIndex << std::endl;;
	// std::cout << typeid(probVector).name() << std::endl;
	int bestAction = mainVector[bestIndex];
	return bestAction;
}
std::pair<torch::Tensor, float> Search::predict_commit(std::unique_ptr<Board>& board) {
	float value;
	torch::Tensor action_probs;
	std::pair< torch::Tensor, std::promise<std::pair<torch::Tensor, torch::Tensor> >  >  message;
	message.first = board->getFeature();
	auto future = message.second.get_future();

	{//task操作域
		std::unique_lock<std::mutex> lock(this->QueueLock);
		this->MessageQueue.push(std::move(message));
	}

	this->asySearchForward.notify_all();

	auto answer = future.get();

	action_probs = std::move(answer.first.flatten());
	value = std::move(answer.second.item().toFloat());

	return  std::pair<torch::Tensor, float>(action_probs, value);
}
void Search::nn_evaluate() 
{
	torch::Tensor action_probs_b;
	torch::Tensor value_b;

	std::unique_lock<std::mutex> lock(this->QueueLock);
	//int a = 0;
	while (!this->finished.load()) 
	{
		std::vector<torch::Tensor> states;
		std::vector<std::promise<std::pair<torch::Tensor, torch::Tensor> > >  promises;

		this->asySearchForward.wait_for(lock, std::chrono::microseconds(this->serverWaitTimeUs), [this] { return this->MessageQueue.size() >= this->BatchSize; });

		//1ms内没有新请求或达到batch_size，将请求推送给nn，否则继续收集数据
		while (states.size() < this->BatchSize && !this->MessageQueue.empty())
		{
			states.emplace_back(std::move(this->MessageQueue.front().first));
			promises.emplace_back(std::move(this->MessageQueue.front().second));
			this->MessageQueue.pop();
		}

		if (states.empty()) continue;

		lock.unlock();

		std::vector<torch::jit::IValue> inputs{ torch::cat(states,0).to(deviceType) };
		auto outputs = this->model.forward(inputs);
		auto output = outputs.toTuple();
		action_probs_b = exp(output->elements()[0].toTensor().data().to(at::kCPU));
		value_b = output->elements()[1].toTensor().data().to(at::kCPU);

		for (int i = 0; i < promises.size(); ++i) {
			std::pair<torch::Tensor, torch::Tensor> answer;
			answer.first = action_probs_b[i];
			answer.second = value_b[i];
			promises[i].set_value(std::move(answer));

		}

		lock.lock();
	}
}

