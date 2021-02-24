#include <fstream>
#include <random>

#include "ETL.h"

Eigen::MatrixXd ETL::get_data_matrix(bool shuffle_rows)
{
	std::vector<std::vector<std::string>> data_array = read_csv();
	int rows = (int)data_array.size();
	int cols = (int)data_array[0].size();

	if (shuffle_rows)
	{
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(data_array.begin(), data_array.end(), g);
	}

	Eigen::MatrixXd mat(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			mat(i, j) = atof(data_array[i][j].c_str());
		}
	}

	return mat;
}

void ETL::splitXY(const Eigen::MatrixXd& data_matrix, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y, const int num_of_dependent_variables)
{
	train_x = data_matrix.leftCols(data_matrix.cols() - num_of_dependent_variables);
	train_y = data_matrix.rightCols(num_of_dependent_variables);
}

void ETL::split_train_test(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y, Eigen::MatrixXd& test_x, Eigen::MatrixXd& test_y, const int test_size)
{
	train_x = data_x.topRows(data_x.rows() - test_size);
	train_y = data_y.topRows(data_y.rows() - test_size);

	test_x = data_x.bottomRows(test_size);
	test_y = data_y.bottomRows(test_size);
}

void ETL::one_hot_encoding(Eigen::MatrixXd& mat, int size)
{
	mat.conservativeResize(mat.rows(), size);
	for (int i = 0; i < mat.rows(); i++)
	{
		int num = (int)mat.row(i)(0);
		for (int j = 0; j < mat.cols(); j++)
		{
			mat.row(i)(j) = 0;
		}
		mat.row(i)(num) = 1;
	}
}

void ETL::replace_zero_one(Eigen::MatrixXd& mat, float replace_zero_to /*= 0.01f*/, float replace_one_to /*= 0.99f*/)
{
	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
		{	
			if (mat.row(i)(j) == 0)
			{
				mat.row(i)(j) = replace_zero_to;
			}
			else if (mat.row(i)(j) == 1)
			{
				mat.row(i)(j) = replace_one_to;
			}
		}
	}
}

int ETL::reverse_int(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ETL::read_MNIST(std::string mnist_image_path, std::string mnist_label_path, int NumberOfImages, int pixel_size, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y)
{
	// train_x(pixel value 28x28)
	train_x = Eigen::MatrixXd(NumberOfImages, pixel_size * pixel_size);
	std::ifstream image_file(mnist_image_path, std::ios::binary);
	if (image_file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		image_file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		image_file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverse_int(number_of_images);
		image_file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverse_int(n_rows);
		image_file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverse_int(n_cols);

		//char inputstring[1000];
		for (int i = 0; i < NumberOfImages; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					image_file.read((char*)&temp, sizeof(temp));
					train_x.row(i)((n_rows * r) + c) = (double)temp;
				}
			}
		}
	}
	image_file.close();

	// train_y(labels)
	train_y = Eigen::MatrixXd(NumberOfImages, 1);
	std::ifstream label_file(mnist_label_path);
	if (label_file.is_open())
	{
		for (int i = 0; i < NumberOfImages + 8; ++i)
		{
			char label = 0;
			label_file.read(&label, 1);
			std::string sLabel = std::to_string(int(label));
			if (i > 7)
			{
				train_y.row(i - 8)(0) = int(label);
			}
		}
	}
	label_file.close();
}

std::vector<std::vector<std::string>> ETL::read_csv()
{
	std::vector<std::vector<std::string>> dataString;
	std::ifstream file(path);
	std::string line = "";
	std::vector<std::string> vec;
	while (getline(file, line))
	{	
		vec.clear();
		std::stringstream ss(line);
		std::string name;
		while (getline(ss, name, separator))
		{
			vec.push_back(name);
		}
		dataString.push_back(vec);
	}
	file.close();

	if (header)
	{
		dataString.erase(dataString.begin());
	}

	return dataString;
}
