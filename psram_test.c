#include <stdio.h>
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_chip_info.h"

void app_main(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);

    printf("\n================ RHEAR D2 MEMORY TEST ================\n");
    printf("CPU frequency: %d MHz\n", esp_clk_cpu_freq() / 1000000);
    printf("Chip cores: %d\n", chip.cores);

    printf("\n--- INTERNAL RAM ---\n");
    printf("Free  : %u bytes (%.2f KB)\n",
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0);

    printf("Largest block: %u bytes (%.2f KB)\n",
           heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024.0);

    printf("\n--- PSRAM ---\n");
    printf("Free  : %u bytes (%.2f KB)\n",
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024.0);

    printf("Largest block: %u bytes (%.2f KB)\n",
           heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
           heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024.0);

    printf("\n--- TOTAL HEAP ---\n");
    printf("Free heap: %u bytes (%.2f KB)\n",
           esp_get_free_heap_size(),
           esp_get_free_heap_size() / 1024.0);

    printf("\n=======================================================\n");
}
