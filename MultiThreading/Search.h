#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include<queue>
#include<atomic>
#include<future>
#include<vector>
#include "Node.h"
#include "Board.h"
#define NOMINMAX
#define numThread 64
#include <windows.h>
#undef NOMINMAX

struct policyValue { //forward返回的结果
	torch::Tensor policy;
	torch::Tensor value;

};
struct stateForward {  //预测请求
	torch::Tensor state;
	std::promise<policyValue> promise;
};

class Search
{
private:
	torch::DeviceType deviceType;
	float cPuct = 1.0f;
	//float cVirtualLoss = 0.0f;
	int BatchSize = 32; //(1+numThread) / 2;
	unsigned SearchTime = 5;
	unsigned serverWaitTimeUs = 1000;
	std::atomic<int> finished;
	std::atomic<int> is_end; //0未结束，1结束
	std::mutex QueueLock;
	std::queue<std::pair< torch::Tensor, std::promise<std::pair<torch::Tensor, torch::Tensor> > > > MessageQueue;
	torch::jit::script::Module model;
	Node* root;
	std::thread nnserver;
public:
	int nodeCount = 0;
	float winRate = 0.0f;
	std::condition_variable asySearchForward; // 推理线程与搜索线程异步
	//int count = 0;
	Search(float cPuct, unsigned ms);
	~Search();
	void getRank(Board* board);
	int getCount() { return nodeCount; }
	float getWinRate() { return winRate; }
	void calculation(const Board* board);
	int getAction(const Board* board);
	std::vector<float> getPi(std::vector<float>& visits, float T);
	void resetRoot();
	void nn_evaluate();
	std::pair<torch::Tensor, float> predict_commit(std::unique_ptr<Board>& board);

	/*
	* 以下方法只是实现search过程中用到的工具方法
	*/
	// 以向量的内容作为下标切片另一个向量
	std::vector<float> getElementsByVector(
		std::vector<float> mainVector,
		std::vector<int> indexVector
	);
	// 主向量按照概率向量的概率分布进行抽样
	int unequalProbSample(
		std::vector<int> mainVector,
		std::vector<float> probVector
	);
	// 推送队列消息
};

