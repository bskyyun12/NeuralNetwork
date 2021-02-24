#pragma once
#include "Layer.h"
#include "Activation.h"

class HiddenLayer : public Layer
{
public:
	HiddenLayer(std::string layer_name, int size)
	{
		name = layer_name;
		layer_vec = Eigen::VectorXd::Ones(size);
	}

	void propagate_forward() override
	{
		if (next_layer->get_is_output_layer()) // next layer is output layer
		{
			//// output sigmoid
			//Eigen::MatrixXd vec = (weight * layer_vec + bias).unaryExpr(Activation::Sigmoid());

			// output softmax 
			Eigen::VectorXd vec = (weight * layer_vec + bias).unaryExpr([](float elem) { return std::exp(elem); });
			vec /= vec.rowwise().sum().sum();

			next_layer->set_layer_vec(vec);
		}
		else // next layer is another hidden layer
		{
			Eigen::MatrixXd next_layer_vec = (weight * layer_vec + bias).unaryExpr(Activation::Sigmoid());
			next_layer->set_layer_vec(next_layer_vec);
		}
	}

	void propagate_backward(const Eigen::VectorXd& target, const float learning_rate) override
	{
		Eigen::MatrixXd hidden = get_layer_vec();

		// get error target - output;
		Eigen::MatrixXd hidden_error = get_weight().transpose() * next_layer->get_error();
		set_error(hidden_error);

		// gradient = (sigmoid_h * (1 - sigmoid_h)) * error * learning_rate
		Eigen::MatrixXd gradient = hidden.unaryExpr(Activation::DerivativeSigmoid()).array() * get_error().array() * learning_rate;

		// delta_weight = gradient dot prev_layer_transpose
		Eigen::MatrixXd prev_layer_transpose = prev_layer->get_layer_vec().transpose();
		Eigen::MatrixXd delta_weight = gradient * prev_layer_transpose;
		//Eigen::MatrixXd delta_weight = gradient * prev_layer_transpose;
		prev_layer->set_weight(prev_layer->get_weight() + delta_weight);

		// delta bias = gradient
		prev_layer->set_bias(prev_layer->get_bias() + gradient);
	}

private:

};

