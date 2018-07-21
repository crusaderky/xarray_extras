/*
 * High speed implementation for to_csv()
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Convert 2D array of doubles to CSV
 *
 * buf : output buffer
 * bufsize : maximum number of characters that can be written to buf
 * array : input numerical data
 * h : number of rows in array
 * w : number of columns in array
 * index : newline-separated list of prefix strings, one per row
 * fmt : printf formatting, including the cell separator at the end.
 *       cell separator must be exactly 1 character.
 * na_rep : string representation for NaN, including the cell separator at the end
 *
 * The line terminator is always \n, regardless of OS.
 */
int snprintcsvd(char * buf, int bufsize, const double * array, int h, int w,
                const char * index, const char * fmt, const char * na_rep)
{
    int char_count = 0;
    int i, j;

    // Move along a single column, printing the value of each row
    for (i = 0; i < h; i++) {
        while (1) {
            char c = *index;
            index++;
            if (c == '\n' || char_count == bufsize) {
                break;
            }
            buf[char_count++] = c;
        }
        for (j = 0; j < w; j++) {
            double n = *array;
            array++;

            if (isnan(n)) {
                char_count += snprintf(buf + char_count, bufsize - char_count, "%s", na_rep);
            }
            else {
                char_count += snprintf(buf + char_count, bufsize - char_count, fmt, n);
            }
        }
        // Replace latest column separator with line terminator
        buf[char_count - 1] = '\n';
    }

    return char_count;
}


/* Convert 2D array of int64's to CSV
 *
 * buf : output buffer
 * bufsize : maximum number of characters that can be written to buf
 * array : input numerical data
 * h : number of rows in array
 * w : number of columns in array
 * index : newline-separated list of prefix strings, one per row
 * sep : cell separator
 *
 * The line terminator is always \n, regardless of OS.
 */
int snprintcsvi(char * buf, int bufsize, const int64_t * array, int h, int w,
                const char * index, char sep)
{
    int char_count = 0;
    int i, j;

    // '%d' + sep, but for int64_t
    char fmt[sizeof(PRId64) + 2];
    fmt[0] = '%';
    strcpy(fmt + 1, PRId64);
    fmt[sizeof(PRId64)] = sep;
    fmt[sizeof(PRId64) + 1] = 0;

    // Move along a single column, printing the value of each row
    for (i = 0; i < h; i++) {
        while (1) {
            char c = *index;
            index++;
            if (c == '\n' || char_count == bufsize) {
                break;
            }
            buf[char_count++] = c;
        }
        for (j = 0; j < w; j++) {
            char_count += snprintf(buf + char_count, bufsize - char_count, fmt, *array);
            array++;
        }
        // Replace latest column separator with line terminator
        buf[char_count - 1] = '\n';
    }

    return char_count;
}