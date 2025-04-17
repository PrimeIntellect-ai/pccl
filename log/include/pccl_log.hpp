#ifndef PCCL_LOG_H
#define PCCL_LOG_H

#include <sstream>

enum LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERR,
    FATAL,
    BUG,
    NONE
};

class NullBuffer final : public std::streambuf {
public:
    int overflow(const int c) override { return c; }
};

class NullStream final : public std::ostream {
public:
    NullStream();

private:
    NullBuffer m_sb;
};

class Logger final {
public:
    Logger();

    virtual ~Logger();

    std::ostream &getStream(LogLevel level);

    static LogLevel &getReportingLevel();

    static void setReportingLevel(LogLevel level);

private:
    Logger(const Logger &);

    Logger &operator=(const Logger &);

    static LogLevel reportingLevel;
    std::ostringstream os;
    LogLevel messageLevel;
};

#define LOG(level) \
Logger().getStream(level)


#endif // PCCL_LOG_H
