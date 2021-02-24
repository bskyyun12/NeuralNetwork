#pragma once
#include "Layer.h"
#include "Activation.h"

class OutputLayer : public Layer
{
public:
	OutputLayer(std::string layer_name, int size)
	{
		name = layer_name;
		layer_vec = Eigen::VectorXd(size);
		is_output_layer = true;
	}

	void propagate_forward() override
	{
		assert("OutputLayer should not call propagate_forward()" && false);
	}

	void propagate_backward(const Eigen::VectorXd& target, const float learning_rate) override
	{
		Eigen::MatrixXd output = get_layer_vec();

		// get error target - output;
		set_error(target - output);

		// gradient = (sigmoid_o * (1 - sigmoid_o)) * error * learning_rate
		Eigen::MatrixXd gradient = output.unaryExpr(Activation::DerivativeSigmoid()).array() * get_error().array() * learning_rate;

		// delta_weight = gradient dot prev_layer_transpose
		Eigen::MatrixXd prev_layer_transpose = prev_layer->get_layer_vec().transpose();
		Eigen::MatrixXd delta_weight = gradient * prev_layer_transpose;

		//Eigen::MatrixXd delta_weight = gradient * prev_layer_transpose * learning_rate;
		prev_layer->set_weight(prev_layer->get_weight() + delta_weight);

		// delta bias = gradient
		prev_layer->set_bias(prev_layer->get_bias() + gradient);
	}

protected:

private:

};


