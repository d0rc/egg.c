#ifndef EGG_UTILS_TERMINAL_H
#define EGG_UTILS_TERMINAL_H

#include <cstdio>
#include <cstdarg>

// Prints a message that overwrites the current line.
// Uses ANSI escape codes to clear the line from the cursor to the end.
inline void print_progress_overwrite(const char* format, ...) {
    va_list args;
    va_start(args, format);
    printf("\r"); // Move to start of line
    vprintf(format, args);
    printf("\033[K"); // Clear to end of line
    fflush(stdout);
    va_end(args);
}

#endif // EGG_UTILS_TERMINAL_H
