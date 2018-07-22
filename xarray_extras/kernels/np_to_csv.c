/*
 * High speed implementation for to_csv()
 */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <Python.h>


#ifdef _WIN32
#    define LIBRARY_API __declspec(dllexport)
#else
#    define LIBRARY_API
#endif


static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC PyInit_np_to_csv(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "np_to_csv",
        "High speed implementation for to_csv()",
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    return module;
}


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
 * trim_zeros : if true, trim trailing zeros after the . beyond the first
 *              e.g. 1.000 -> 1.0
 * na_rep : string representation for NaN, including the cell separator at the end
 *
 * The line terminator is always \n, regardless of OS.
 */
LIBRARY_API
int snprintcsvd(char * buf, int bufsize, const double * array, int h, int w,
                const char * index, const char * fmt, bool trim_zeros, const char * na_rep)
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
                if (trim_zeros) {
                    while (char_count > 2 &&
                           buf[char_count - 2] == '0' &&
                           buf[char_count - 3] != '.') {
                        buf[char_count - 2] = buf[char_count - 1];
                        char_count--;
                    }
                }
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
LIBRARY_API
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