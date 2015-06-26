#include <iostream>
#include <vector>

typedef std::vector<float> VF;

const bool DEBUG = true;
const float learningRate = 1;

void printVec(VF vec) {
	std::cout << "{ ";
	for (int i = 0; i < vec.size(); i++) {
		std::cout << vec[i];
		if (i != vec.size() - 1) std::cout << ", ";
	}
	std::cout << " }" << std::endl;
}

class Vector {
public:
	Vector(){};
	~Vector(){};

	Vector(VF vec) { this->vec = vec; };

	VF vec;

	// Vector contents added together
	float total() {
		float output = 0;
		for (int i = 0; i < vec.size(); i++) {
			output += vec[i];
		}
		return output;
	}

	Vector& operator=(const Vector& other) {
		this->vec = other.vec;
		return *this;
	};

	Vector operator+(const Vector&other) {
		VF tmp;
		for (int i = 0; i < vec.size(); i++) {
			tmp.push_back(vec[i] + other.vec[i]);
		}

		return Vector(tmp);
	};

	Vector operator*(const Vector& other) {
		VF tmp;
		for (int i = 0; i < vec.size(); i++) {
			tmp.push_back(vec[i] * other.vec[i]);
		}

		return Vector(tmp);
	};

	Vector operator*(const float& constant) {
		VF tmp;
		for (int i = 0; i < vec.size(); i++) {
			tmp.push_back(vec[i] * constant);
		}

		return Vector(tmp);
	};
};

class Neuron {
public:
	Neuron() { 
		float arr[] = { 0, 0, 0 };	
		VF tmp(arr, arr + sizeof(arr) / sizeof(float));

		weights = Vector(tmp);
	};
	~Neuron(){};

	Neuron(Vector weights) { this->weights = weights; };

	// Bias is built in at weights[0]
	Vector weights;

	void recalculateWeights(float learningRate, int actualOutput, int desiredOutput, Vector &input) {
		if (DEBUG) {
			std::cout << "Recalculating weights based on: ";
			printVec(input.vec);
			std::cout << "Original Weights: ";
			printVec(weights.vec);	
		}
		// w(n+1) = w(n) + (x(n) * (lR * (d(n) - y(n))))
		weights = (weights + (input * (learningRate * (desiredOutput - actualOutput))));	
		if (DEBUG) {
			std::cout << "Recalculated Weights: ";
			printVec(weights.vec);
		}
	};

	float summationOfWeightedInput(Vector &input) {
		return (weights * input).total();
	};

	int classify(Vector &input) {
		float total = summationOfWeightedInput(input);

		if (DEBUG) std::cout << "Summation: " << total << std::endl;

		if (total < 0) return -1;
		else return 1;
	};
};

class Data {
public:
	Data(){};
	~Data(){};

	Data(float p1, float p2, int classification) {
		float arr[] = { 1, p1, p2 };
		VF tmp(arr, arr + sizeof(arr) / sizeof(float));

		input = Vector(tmp);	

		this->classification = classification;
	};

	int classification;
	Vector input;
};

void train(std::vector<Data> knowndata, Neuron &neuron) {
	int currentTrainingIndex = 0;
	int dataSize = knowndata.size();

	for (int numLeft = dataSize; numLeft != 0; numLeft--) {
		if (DEBUG) {
			std::cout << "===============================================" << std::endl;
			std::cout << "Currently Training: " << currentTrainingIndex << std::endl;
		}
		Data curr = knowndata[currentTrainingIndex];
		int classificationGuess = neuron.classify(curr.input);
		if (classificationGuess != curr.classification) {
			// Reweight based on data
			neuron.recalculateWeights(learningRate, classificationGuess, curr.classification, curr.input);
			numLeft = dataSize + 1; // Resets so that it cycles through to recheck everything
		}

		if (currentTrainingIndex < dataSize - 1) currentTrainingIndex++;
		else currentTrainingIndex = 0;
		std::cout << std::endl;
	}
}

int main() {
	// Initialization of Known Data
	std::vector<Data> knownData;
	knownData.push_back(Data(121, 16.8, 1));
	knownData.push_back(Data(114, 15.2, 1));	
	knownData.push_back(Data(210, 9.4, -1));
	knownData.push_back(Data(195, 8.1, -1));


	float arr1[] = { 1, 140, 17.9 };
	VF vec(arr1, arr1 + sizeof(arr1) / sizeof(float));
	Vector unknown(vec);

	float arr2[] = { -1230, -30, 300 };
	VF knownWeights(arr2, arr2 + sizeof(arr2) / sizeof(float));
	Neuron neuron;

	std::cout << "Learning Rate: " << learningRate << std::endl << std::endl;

	train(knownData, neuron);

	std::cout << "===============================================" << std::endl;
	std::cout << "Classifying unknown: " << std::endl;
	std::cout << "Classification of Unknown: " << neuron.classify(unknown) << std::endl;

	return 0;
}
