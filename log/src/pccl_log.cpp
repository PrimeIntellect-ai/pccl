#include "pccl_log.hpp"

#include <cstring>
#include <ctime>
#include <iostream>

const char *ToString(const LogLevel level) {
    switch (level) {
        case DEBUG:
            return "DEBUG";
        case INFO:
            return "INFO";
        case WARN:
            return "WARN";
        case ERR:
            return "ERR";
        case FATAL:
            return "FATAL";
        case BUG:
            return "BUG";
        default:
            return "UNKNOWN";
    }
}

static LogLevel readReportingLevel() {
    const char *logLevel = getenv("PCCL_LOG_LEVEL");

    if (logLevel == nullptr) {
        return NONE;
    }
    if (strcmp(logLevel, "DEBUG") == 0) {
        return DEBUG;
    }
    if (strcmp(logLevel, "INFO") == 0) {
        return INFO;
    }
    if (strcmp(logLevel, "WARN") == 0) {
        return WARN;
    }
    if (strcmp(logLevel, "ERR") == 0) {
        return ERR;
    }
    if (strcmp(logLevel, "FATAL") == 0) {
        return FATAL;
    }
    if (strcmp(logLevel, "NONE") == 0) {
        return NONE;
    }
    return NONE;
}

LogLevel Logger::reportingLevel = readReportingLevel();

NullStream::NullStream(): std::ostream(&m_sb) {
}

Logger::Logger() : messageLevel(INFO) {
}

Logger::~Logger() {
    if (messageLevel >= reportingLevel) {
        os << std::endl;
        std::cout << os.str();
        if (messageLevel == FATAL || messageLevel == BUG) {
            LOG(INFO) << "Logged fatal/bug message; exiting...";
            exit(1);
        }
    }
}

std::ostream &Logger::getStream(const LogLevel level) {
    messageLevel = level;

    if (level >= reportingLevel) {
        // Get current time
        time_t raw_time;
        char buffer[55];
        time(&raw_time);
        const tm *time_info = localtime(&raw_time);
        strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", time_info);

        os << buffer << " - " << ToString(level) << ": ";

        return os;
    }

    static NullStream null_stream;
    return null_stream;
}

LogLevel &Logger::getReportingLevel() {
    return reportingLevel;
}

void Logger::setReportingLevel(const LogLevel level) {
    reportingLevel = level;
}
