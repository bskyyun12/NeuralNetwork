#pragma once
#include "Layer.h"
#include "Activation.h"

class InputLayer : public Layer
{
public:
	InputLayer(std::string layer_name, int size)
	{
		name = layer_name;
		layer_vec = Eigen::VectorXd(size);
	}

	void propagate_forward() override
	{
		Eigen::MatrixXd next_layer_vec = (weight * layer_vec + bias).unaryExpr(Activation::Sigmoid());
		next_layer->set_layer_vec(next_layer_vec);
	}

	void propagate_backward(const Eigen::VectorXd& target, const float learning_rate) override
	{
		assert("InputLayer should not call propagate_backward()" && false);
	}

protected:

private:

};

