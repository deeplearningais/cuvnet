 // Sample Program:
 // Compiler: gcc 3.2.3 20030502
 // Linux: Red Hat

 #include <execinfo.h>
 #include <signal.h>

 #include <exception>
 #include <iostream>

 /////////////////////////////////////////////

 class ExceptionTracer
 {
 public:
     ExceptionTracer()
     {
         void * array[25];
         int nSize = backtrace(array, 25);
         char ** symbols = backtrace_symbols(array, nSize);

         for (int i = 0; i < nSize; i++) {
             std::cout << symbols[i] << std::endl;
         }

         free(symbols);
     }
 };

template <class SignalExceptionClass> class SignalTranslator {
    private:
        class SingleTonTranslator {
            public:
                SingleTonTranslator() {
                    signal(SignalExceptionClass::GetSignalNumber(), SignalHandler);
                }
                static void SignalHandler(int) {
                    s_func();
                    //throw SignalExceptionClass();
                }
        };

        static SingleTonTranslator s_objTranslator;
        static boost::function<void(void)> s_func;
    public:
        SignalTranslator(const boost::function<void(void)>& f)
        {
            s_func = f;
        }
        ~SignalTranslator()
        {
            s_func.clear();
        }
};

template<class SignalExceptionClass>
typename SignalTranslator<SignalExceptionClass>::SingleTonTranslator 
SignalTranslator<SignalExceptionClass>::s_objTranslator;

template<class SignalExceptionClass>
boost::function<void(void)> 
SignalTranslator<SignalExceptionClass>::s_func;

// An example for SIGINT
struct CtrlCPressed : public ExceptionTracer, public std::exception {
        static int GetSignalNumber() {return SIGINT;}
};

