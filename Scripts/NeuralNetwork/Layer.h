#pragma once
#include <eigen3/Eigen/Dense>
#include <random>

class Layer
{
public:
	~Layer()
	{}

	virtual void propagate_forward() = 0;
	virtual void propagate_backward(const Eigen::VectorXd& target, const float learning_rate) = 0;

	// getters
	const std::string get_name() { return name; }
	const Eigen::VectorXd get_layer_vec() { return layer_vec; }

	const int get_num_of_neurons() { return (int)layer_vec.size(); }
	const Eigen::MatrixXd get_weight() { return weight; }
	const Eigen::VectorXd get_bias() { return bias; }
	const Eigen::MatrixXd get_error() { return error; }
	const std::string get_weight_bias_name() { return weight_bias_name; }
	const bool get_is_output_layer() { return is_output_layer; }


	// setters
	void set_layer_vec(const Eigen::VectorXd& layer) { layer_vec = layer; }
	void set_next_layer(const std::shared_ptr<Layer>& layer) { next_layer = layer; }
	void set_prev_layer(const std::shared_ptr<Layer>& layer) { prev_layer = layer; }

	void set_weight(const Eigen::MatrixXd& w) { weight = w; }
	void set_bias(const Eigen::VectorXd& b) { bias = b; }
	void set_error(const Eigen::MatrixXd& e) { error = e; }

	void init_weight_bias(const std::string& name)
	{
		weight_bias_name = name;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-.1f, 0.f);

		weight = Eigen::MatrixXd::Ones(next_layer->get_num_of_neurons(), layer_vec.rows()).unaryExpr([&](float elem) {return elem = dis(gen); });
		bias = Eigen::VectorXd::Ones(next_layer->get_num_of_neurons()).unaryExpr([&](float elem) {return elem = dis(gen); });

		//std::cout << "--------------" << std::endl;
		//std::cout << weight_bias_name << std::endl;
		//std::cout << "weight: " << std::endl;
		//std::cout << weight << std::endl;
		//std::cout << "bias: " << std::endl;
		//std::cout << bias << std::endl;
		//std::cout << "--------------" << std::endl;
	}

protected:
	std::string name;
	Eigen::VectorXd layer_vec;

	std::shared_ptr<Layer> prev_layer;
	std::shared_ptr<Layer> next_layer;

	std::string weight_bias_name;
	Eigen::MatrixXd weight;
	Eigen::VectorXd bias;

	Eigen::MatrixXd error;

	bool is_output_layer;

private:

};