#include "pch.h"
#include "CppUnitTest.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTests
{
	TEST_CLASS(LP_TRANSFORMER)
	{
	public:
		
		TEST_METHOD(TEST_1_BASIC)
		{
			/**
			*  Transform the following LP into standard form using the BasicTransformer
			*  
			*    min [1,2]^T x
			*    s.t.
			*    ┌─   ─┐      ┌─ ─┐
			*    │ 1 2 │ x =  │ 5 │
			*    │ 3 4 │      │ 6 │
			*    └─   ─┘      └─ ─┘
			*    ┌─   ─┐       ┌─ ─┐
			*    │ 7 8 │ x <=  │ 1 │
			*    │ 9 0 │       │ 2 │
			*    └─   ─┘       └─ ─┘
			* 
			*	 x[0] <= 1
			*    x[1] >= 3
			*/
			

		}
	};
}
