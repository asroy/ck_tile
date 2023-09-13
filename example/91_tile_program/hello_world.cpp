#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

#include "tile_program.hpp"

#include "ck/library/utility/device_memory.hpp"

// ProgramServer contains a "meta data buffer"
// host evaluate the expression inside ps(), and push the result into meta data buffer
// ProgramServer send meta data buffer to GPU as kernel arguement
// device read (not evaluate) the value of the expression inside ps() from meta data buffer
struct HelloWorld
{
    __host__ __device__ void operator()(ProgramServer& ps, int x, int y, int* res)
    {
        auto r0 = ps(x + y);
        auto r1 = ps(x - y);

        res[0] = r0;
        res[1] = r1;
    }
};

int main()
{
    int x = 100;
    int y = 101;

    DeviceMem res_dev_buf(2 * sizeof(int));

    launch(ProgramServer{},
           HelloWorld{},
           1,
           64,
           x,
           y,
           static_cast<int*>(res_dev_buf.GetDeviceBuffer()));

    int res_host[2];

    res_dev_buf.FromDevice(&res_host);

    printf("x+y=: %d\n", res_host[0]);
    printf("x-y=: %d\n", res_host[1]);

    return 0;
}
