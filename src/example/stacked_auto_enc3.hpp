#ifndef __STACKED_AUTO_ENC3_HPP__
#     define __STACKED_AUTO_ENC3_HPP__

#include <boost/program_options.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>                                                                  
#include <sys/syscall.h> /* for pid_t, syscall, SYS_gettid */

#include <mongo/client/dbclient.h>
#include <mdbq/client.hpp>

#include <cuvnet/ops.hpp>
#include <cuvnet/op_utils.hpp>
#include <cuvnet/models/auto_encoder_stack.hpp>
#include <cuvnet/models/logistic_regression.hpp>
#include <cuvnet/models/contractive_auto_encoder.hpp>
#include <tools/gradient_descent.hpp>
#include <tools/crossvalid.hpp>
#include <tools/learner.hpp>
#include <tools/monitor.hpp>
#include <tools/network_communication.hpp>
#include <tools/logging.hpp>
#include <tools/python_helper.hpp>

#endif /* __STACKED_AUTO_ENC3_HPP__ */
