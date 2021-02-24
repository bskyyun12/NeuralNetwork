#pragma once
#include <vector>

#include "Layer.h"

class NeuralNetwork
{
public:
	NeuralNetwork() {}

	~NeuralNetwork()
	{}

	void add_layer(std::unique_ptr<Layer> new_layer)
	{
		std::shared_ptr<Layer> shared = std::move(new_layer);
		// set up 'next layer' and 'prev layer'
		if (layers.size() != 0)
		{
			shared->set_prev_layer(layers.back());
			layers.back()->set_next_layer(shared);
		}

		layers.push_back(shared);
	}

	void make_model()
	{
		// set up weight and bias
		for (int i = 0; i < layers.size() - 1; i++)
		{
			std::string name = layers[i]->get_name() + " to " + layers[i + 1]->get_name();
			layers[i]->init_weight_bias(name);
		}
	}

	void train(const Eigen::MatrixXd& train_x, const Eigen::MatrixXd& train_y, const float learning_rate)
	{
		for (int i = 0; i < train_x.rows(); i++)
		{
			propagate_forward(train_x.row(i).transpose());
			propagate_backward(train_y.row(i).transpose(), learning_rate);
		}
	}

	void predict(const Eigen::MatrixXd& test_x, const Eigen::MatrixXd& test_y)
	{
		// categorical_crossentropy: -sum(answer * log(guess))
		float categorical_crossentropy = 0.f;
		float sum_of_categorical_crossentropy = 0.f;
		int correct_guesses = 0;
		for (int i = 0; i < test_x.rows(); i++)
		{
			propagate_forward(test_x.row(i).transpose());

			Eigen::VectorXd guess = get_output_vector();
			Eigen::VectorXd answer = test_y.row(i);

			// SoftMax
			guess = guess.unaryExpr([](double elem) { return std::exp(elem); });
			guess /= guess.rowwise().sum().sum();

			// categorical_crossentropy: -sum(answer * log(guess))
			categorical_crossentropy = -1 * (float)(answer.transpose() * (guess.unaryExpr([](float x) {return std::log(x); }))).sum();
			sum_of_categorical_crossentropy += categorical_crossentropy;

			//std::cout << "Inputs: \n" << test_x.row(i) << std::endl;
			int answer_index = get_index_of_max_value(answer);
			int guess_index = get_index_of_max_value(guess);

			//std::cout << "Test " << i << ". " << "Answer: " << answer_index << ", guessed: " << guess_index;
			if (answer_index == guess_index)
			{
				correct_guesses++;
				//std::cout << " (Good!)" << std::endl;
			}
			else
			{
				//std::cout << " (Baad!)" << std::endl;
				//std::cout << "guess: (" << guess.transpose() << ")" << std::endl;
				//std::cout << "answer: (" << answer.transpose() << ")" << std::endl;
			}
			//std::cout << "-----------------------------" << std::endl;
		}
		std::cout << "-----------------------------" << std::endl;
		std::cout << "Prediction result: " << correct_guesses << "/" << test_x.rows() << std::endl;
		std::cout << "categorical_crossentropy: " << categorical_crossentropy / test_x.rows() << std::endl;
		std::cout << "-----------------------------" << std::endl;
	}

	int get_index_of_max_value(Eigen::VectorXd vec)
	{
		for (int i = 0; i < vec.size(); i++)
		{
			if (vec[i] == vec.maxCoeff())
			{
				return i;
			}
		}

		return -1;
	}

	void propagate_forward(const Eigen::VectorXd& input_layer)
	{
		layers[0]->set_layer_vec(input_layer);

		for (int i = 0; i < layers.size() - 1; i++)
		{
			layers[i]->propagate_forward();
		}
	}

	void propagate_backward(const Eigen::VectorXd& target, const float learning_rate)
	{
		for (int i = (int)layers.size() - 1; i > 0; i--)
		{
			layers[i]->propagate_backward(target, learning_rate);
		}
	}

	// Debugging methods
	void pad_to(std::string& str, const int num)
	{
		if (num > str.size())
		{
			str.insert(0, num - str.size(), ' ');
		}
	}
	void summary()
	{
		std::vector<std::vector<std::string>> summary;
		summary.push_back({ "Layer(type)", "Output Shape", "Param #" });

		int total_params = 0;
		for (int i = 0; i < layers.size(); i++)
		{
			int prev_layer_vec_rows = i == 0 ? -1 : layers[i - 1]->get_num_of_neurons();
			int layer_vec_rows = layers[i]->get_num_of_neurons();
			int params = (prev_layer_vec_rows + 1) * layer_vec_rows;
			total_params += params;

			std::string layer_name = layers[i]->get_name();
			std::string shape = "(" + std::to_string(layers[i]->get_num_of_neurons()) + "," + std::to_string(layers[i]->get_layer_vec().cols()) + ")";
			std::string params_s = std::to_string(params);

			pad_to(shape, 15);
			pad_to(params_s, 9);

			summary.push_back({ layer_name, shape, params_s });
		}

		std::cout << "------------------------------------------" << std::endl;
		std::cout << "------------------------------------------" << std::endl;
		for (int i = 0; i < summary.size(); i++)
		{
			for (int j = 0; j < summary[0].size(); j++)
			{
				std::cout << summary[i][j] << "     ";
			}
			std::cout << std::endl;
		}
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "total_params: " << total_params << std::endl;
		std::cout << "------------------------------------------" << std::endl;
		std::cout << "------------------------------------------" << std::endl;
	}
	const Eigen::VectorXd get_output_vector() { return layers[layers.size() - 1]->get_layer_vec(); }

private:
	std::vector<std::shared_ptr<Layer>> layers;
};

