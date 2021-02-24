#pragma once
#include <iostream>

namespace Activation
{
	float sigmoid(float x)
	{
		return 1.f / (1.f + std::exp(-x));
	}

	float swish(float x)
	{
		return x * sigmoid(x);
	}

	struct Sigmoid
	{
		float operator()(float x) const
		{
			return sigmoid(x);
		}
	};

	struct DerivativeSigmoid
	{
		float operator()(float sigmoid_x) const
		{
			return (float)(sigmoid_x * (1.f - sigmoid_x));
		}
	};

	struct Swish
	{
		float operator()(float x) const
		{
			return swish(x);
		}
	};

	struct DerivativeSwish
	{
		float operator()(float x) const
		{
			return (float)(swish(x) + sigmoid(x) * (1 - swish(x)));
		}
	};
}

