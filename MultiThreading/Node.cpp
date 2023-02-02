#include "Node.h"
#include "Search.h"

// 构造函数
Node::Node(float priorProb, float cPuct, Node* parent)
{
	this->Q = 0.0f;
	//this->U = 0.0;
	this->N = 0;
	this->P = priorProb;
	this->cPuct = cPuct;
	this->parent = parent;
}

// 析构函数
Node::~Node()
{
	for (auto it : children)
		delete it.second;
}

// 选择得分最高的节点
// 返回std::pair<int, Node*>(maxAction, maxNode)
std::pair<int, Node*> Node::select()
{
	std::shared_lock<std::shared_mutex> lk(nodeLock);

	int maxAction = -1;
	float score = -10000;
	Node* maxNode = nullptr;

	for (std::map<int, Node*>::const_iterator it = children.begin(); it != children.end(); it++)
	{
		float newScore = it->second->getScore();
		if (newScore > score)
		{
			maxAction = it->first;
			maxNode = it->second;
			score = newScore;
		}
	}
	// 增加虚拟损失
	if (maxNode)
		maxNode->virtualLoss++;

	//for (std::map<int, Node*>::iterator it = children.begin();
	//	it != children.end(); it++)
	//{
	//	if (it->first == maxAction)
	//	{
	//		it->second->virtualLoss++;
	//	}
	//}
	return std::pair<int, Node*>(maxAction, maxNode);
}

// 扩展节点
// 传入std::map<int, double>
void Node::expand(const std::map<int, float>& actionProb)
{
	// 外部加锁
	if (children.empty())
	{
		for (std::map<int, float>::const_iterator it = actionProb.begin();
			it != actionProb.end(); it++)
		{
			Node* childNode = new Node(it->second, cPuct, this);
			children.insert(std::pair<int, Node*>(it->first, childNode));
		}
	}
}

// 更新 Q和N
void Node::update(float value)
{
	// 加自旋锁操作，短期操作，不sleep
	std::lock_guard<SpinLock> lk(spinLock);

	virtualLoss--;
	Q = (N * Q + value) / (N + 1);
	N++;
}

// 反向传播
void Node::backup(float value)
{
	if (parent)
	{
		parent->backup(-value);
	}
	update(value);
}

// 获取得分
// 返回score
float Node::getScore()
{
	std::lock_guard<SpinLock> lk(spinLock);

	if (parent == nullptr)
		return 0;

	float U = cPuct * P * sqrtf(parent->N) / (N + 1.0f + virtualLoss * cVirtualLoss);

	float QQ = N ? Q : -(P * 2.0f - 1.0f); //std::max(-1.0f, std::min(1.0f, -parent->Q + cVirtualQDelta));
	int NN = std::max<int>(N, 1);
	float virtualQ = (QQ * NN + std::max(QQ, -parent->Q) * virtualLoss * cVirtualLoss) / (NN + virtualLoss * cVirtualLoss);
	// 注意 Q是对手胜率
	return U - virtualQ;
}

float Node::getUpQ() const
{
	return (-Q * N + (P * 2.0f - 1.0f)) / (N + 1);
}

// 是否为叶子节点
bool Node::isLeafNode(bool lock)
{
	if (lock) nodeLock.lock_shared();
	bool flag = children.empty();
	if (lock) nodeLock.unlock_shared();
	return flag;
}




