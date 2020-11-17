/*
 * Tries out dlopen and my custom cuDNN library.
 * 
 * Remember to link against `libdl` by adding `-ldl`.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <link.h>
#include <dlfcn.h>

#include "cudnn.h"


int main() {
    // MARK: Get environment variables
    printf("PATH: %s\n", getenv("PATH"));
    printf("LD_LIBRARY_PATH: %s\n", getenv("LD_LIBRARY_PATH"));

    // MARK: dlopen and dlinfo
    void* handle = dlopen("libcudnn.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        printf("Failed!\n");
    }

    printf("Success!\n");

    char* origin = malloc(sizeof(char) * (PATH_MAX + 1));
    dlinfo(handle, RTLD_DI_ORIGIN, origin);
    printf("Library path: %s\n", origin);

    struct link_map* m = NULL;
    dlinfo(handle, RTLD_DI_LINKMAP, &m);
    printf("Library path: %s\n", m->l_name);

    dlclose(handle);

    return 0;
}
