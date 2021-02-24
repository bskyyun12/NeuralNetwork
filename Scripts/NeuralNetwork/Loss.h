#pragma once

template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e)
{
	return static_cast<typename std::underlying_type<E>::type>(e);
}

enum class Loss
{
	Categorical_Crossentropy,
	Crossentropy
};

static const char* loss_str[] =
{
	"Categorical_Crossentropy",
	"Crossentropy"
};

static const std::string loss_to_string(Loss loss)
{
	return loss_str[to_underlying(loss)];
}