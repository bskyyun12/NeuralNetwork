#include <eigen3/Eigen/Dense>
#include <iostream>

#include "ETL.h"
#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Layer.h"
#include "NeuralNetwork/InputLayer.h"
#include "NeuralNetwork/HiddenLayer.h"
#include "NeuralNetwork/OutputLayer.h"

void print_matrix(std::string name, Eigen::MatrixXd mat, bool print_mat)
{
	std::cout << name << " => Shape: (" << mat.rows() << "," << mat.cols() << ")" << std::endl;
	if (print_mat)
	{
		std::cout << mat << std::endl << std::endl;
	}
}

int main()
{
#pragma region IRIS
	//ETL etl("Data/iris_data.csv", ',', true);
	//Eigen::MatrixXd data_xy = etl.get_data_matrix();
	//Eigen::MatrixXd data_x, data_y, train_x, train_y, test_x, test_y;
	//etl.splitXY(data_xy, data_x, data_y, 1);
	//etl.one_hot_encoding(data_y, 3);
	//etl.split_train_test(data_x, data_y, train_x, train_y, test_x, test_y, (int)(data_x.rows() * 0.1f));

	//NeuralNetwork nn;
	//nn.add_layer(std::make_unique<InputLayer>(" Input", data_x.cols()));
	//nn.add_layer(std::make_unique<HiddenLayer>("Hidden", 3));
	//nn.add_layer(std::make_unique<OutputLayer>("Output", data_y.cols()));
	//nn.make_model();

	//nn.summary();

	//int epochs = 3000;
	//int print_step = 10;
	//float learning_rate = 0.01f;
	//for (int i = 0; i <= epochs; i++)
	//{
	//	nn.train(train_x, train_y, learning_rate);

	//	if (i % print_step == 0)
	//	{
	//		nn.predict(test_x, test_y);
	//		std::cout << "Trained " << i + 1 << "." << std::endl;
	//	}
	//}
#pragma endregion

#pragma region MNIST TRAIN
	int num_of_images = 300;
	int pixel_size = 28;
	int output_neurons = 10;
	int test_size = 20;

	ETL mnist_etl;
	Eigen::MatrixXd data_x, data_y, train_x, train_y, test_x, test_y;
	mnist_etl.read_MNIST("Data/mnist_images.idx3-ubyte", "Data/mnist_labels.idx1-ubyte", num_of_images, pixel_size, data_x, data_y);
	mnist_etl.one_hot_encoding(data_y, output_neurons);
	data_x /= 255.f;
	mnist_etl.split_train_test(data_x, data_y, train_x, train_y, test_x, test_y, test_size);

	NeuralNetwork nn;
	nn.add_layer(std::make_unique<InputLayer>(" Input", data_x.cols()));
	nn.add_layer(std::make_unique<HiddenLayer>("Hidden", 84));
	nn.add_layer(std::make_unique<OutputLayer>("Output", data_y.cols()));
	nn.make_model();
	nn.summary();

	//print_matrix("data_x", data_x, false);
	//print_matrix("data_y", data_y, false);
	//print_matrix("train_x", train_x, false);
	//print_matrix("train_y", train_y, false);
	//print_matrix("test_x", test_x, true);
	//print_matrix("test_y", test_y, false);


	int epochs = 50;
	int print_step = 1;
	float learning_rate = 0.06f;
	for (int i = 0; i <= epochs; i++)
	{
		nn.train(train_x, train_y, learning_rate);

		if (i % print_step == 0)
		{
			nn.predict(test_x, test_y);
			std::cout << "Trained " << i + 1 << "." << std::endl;
		}
	}

#pragma endregion
}
