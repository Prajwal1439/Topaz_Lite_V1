/* execstack - tool to set or clear the executable stack flag of ELF binaries */
/* from the prelink toolset - open-sourced under GPL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <elf.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

void usage(const char *progname) {
    fprintf(stderr, "Usage: %s [-q] [-s|-c] <elf-binary>...\n", progname);
    fprintf(stderr, "  -s : set executable stack\n");
    fprintf(stderr, "  -c : clear executable stack\n");
    fprintf(stderr, "  -q : query stack executable flag\n");
    exit(1);
}

int main(int argc, char **argv) {
    int set_flag = -1;  // -1: query, 0: clear, 1: set
    int quiet = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-s")) set_flag = 1;
        else if (!strcmp(argv[i], "-c")) set_flag = 0;
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else if (argv[i][0] == '-') usage(argv[0]);
        else break;
    }

    if (i == argc) usage(argv[0]);

    for (; i < argc; i++) {
        const char *filename = argv[i];
        int fd = open(filename, O_RDWR);
        if (fd < 0) {
            perror("open");
            continue;
        }

        struct stat st;
        if (fstat(fd, &st) < 0) {
            perror("fstat");
            close(fd);
            continue;
        }

        void *map = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            perror("mmap");
            close(fd);
            continue;
        }

        Elf64_Ehdr *ehdr = (Elf64_Ehdr *)map;
        Elf64_Phdr *phdr = (Elf64_Phdr *)((char *)map + ehdr->e_phoff);
        int found = 0;

        for (int j = 0; j < ehdr->e_phnum; j++) {
            if (phdr[j].p_type == PT_GNU_STACK) {
                found = 1;
                if (set_flag == -1) {
                    // Query
                    printf("%c %s\n", (phdr[j].p_flags & PF_X) ? 'X' : '-', filename);
                } else if (set_flag == 1) {
                    phdr[j].p_flags |= PF_X;
                } else {
                    phdr[j].p_flags &= ~PF_X;
                }
                break;
            }
        }

        if (!found && !quiet)
            fprintf(stderr, "%s: no PT_GNU_STACK program header\n", filename);

        munmap(map, st.st_size);
        close(fd);
    }

    return 0;
}
