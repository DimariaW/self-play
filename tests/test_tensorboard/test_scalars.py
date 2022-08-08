
import tensorboardX

sw = tensorboardX.SummaryWriter(logdir="./log")
for i in range(1000):

    #sw.add_scalars("main_tag", {"index": i, "index^2": i**2, "index^3": i**3}, global_step=i)
    sw.add_scalar("main/index", i, global_step=i)
    sw.add_scalar("main/index2", i**2, global_step=i)
    sw.add_scalar("main/index3", i**3, global_step=i)

sw.close()
