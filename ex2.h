///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <memory>


/* Abstract base class for both parts of the exercise */
class gpu_image_processing_context
{
public:
    virtual ~gpu_image_processing_context() {}

    /* Enqueue an image for processing. Receives pointers to pinned host
     * memory. Return false if there is no room for image (caller will try again).
     */
    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) = 0;

    /* Checks whether an image has completed processing. If so, set img_id
     * accordingly, and return true. */
    virtual bool dequeue(int *img_id) = 0;
};

std::unique_ptr<gpu_image_processing_context> create_queues_server(int threads);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

