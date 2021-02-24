#include <eigen3/Eigen/Dense>
#include <vector>

class ETL
{
public:
	ETL() {}

	ETL(std::string _path, char _separator, bool _header) : path(_path), separator(_separator), header(_header)
	{}
	
	Eigen::MatrixXd get_data_matrix(bool shuffle_rows = true);

	void splitXY(const Eigen::MatrixXd& data_matrix, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y, const int num_of_dependent_variables);
	void split_train_test(const Eigen::MatrixXd& data_x, const Eigen::MatrixXd& data_y, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y, Eigen::MatrixXd& test_x, Eigen::MatrixXd& test_y, const int test_size);
	void one_hot_encoding(Eigen::MatrixXd& mat, int size);
	void replace_zero_one(Eigen::MatrixXd& mat, float replace_zero_to = 0.01f, float replace_one_to = 0.99f);

	// MNIST
	void read_MNIST(std::string mnist_image_path, std::string mnist_label_path, int NumberOfImages, int pixel_size, Eigen::MatrixXd& train_x, Eigen::MatrixXd& train_y);

private:
	std::string path;
	char separator = ',';
	bool header = true;

	std::vector<std::vector<std::string>> read_csv();

	// MNIST
	int reverse_int(int i);
};