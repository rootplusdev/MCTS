#pragma once
#include"SpinLock.h"
#include<map>
#include<tuple>
#include <atomic>
#include <mutex>
#include <shared_mutex>

class Node
{
private:
	float cPuct = 1.0;
	float cVirtualLoss = 1.0f;
	float cVirtualQDelta = 0.1f;
	Node* parent = nullptr;
public:
	std::atomic<int> N = 0;
	std::atomic<int> virtualLoss = 0;
	float Q = 0.0f;
	//float U = 0.0;
	float P = 1.0f;
	SpinLock spinLock;
	std::shared_mutex nodeLock;
	std::map<int, Node*> children;
	Node(float priorProb, float cPuct, Node* parent);
	~Node();
	std::pair<int, Node*> select();
	void expand(const std::map<int, float>& actionProb);
	void update(float value);
	void backup(float value);
	float getScore();
	float getUpQ() const;
	bool isLeafNode(bool lock = true);
};

