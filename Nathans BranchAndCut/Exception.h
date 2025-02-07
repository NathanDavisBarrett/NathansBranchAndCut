#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <string>
#include <exception>

class ModelFormattingException : public std::exception {
private:
    std::string message;

public:
    // Constructor accepts a const char* that is used to set
    // the exception message
    ModelFormattingException(const char* msg)
        : message(msg)
    {
    }

    // Override the what() method to return our message
    const char* what() const throw()
    {
        return message.c_str();
    }
};

#endif