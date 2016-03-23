#ifndef dbrew_macro_hpp
#define dbrew_macro_hpp

#include <functional>

#define DBREW_SPEC_START \
	{                    \
	std::function<void()> temp = [&]() {

#define DBREW_SPEC_END \
	}                  \
	;                  \
	temp();            \
	/* DBREW MAGIC */  \
	}

#endif /* end of include guard: dbrew_macro_hpp */
