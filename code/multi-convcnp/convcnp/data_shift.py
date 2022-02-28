import lab.torch as B

import stheno

import torch



__all__ = ["GPGeneratorShiftedClassification"]





class GPGeneratorShiftedClassification:

    """GP generator.



    Args:

        kernel (:class:`stheno.Kernel`, optional): Kernel of the GP. Defaults to an

            EQ kernel with length scale `0.25`.

        noise (float, optional): Observation noise. Defaults to `5e-2`.

        seed (int, optional): Seed. Defaults to `0`.

        batch_size (int, optional): Batch size. Defaults to 16.

        num_tasks (int, optional): Number of tasks to generate per epoch. Must be an

            integer multiple of `batch_size`. Defaults to 2^14.

        x_range (tuple[float, float], optional): Range of the inputs. Defaults to

            [-2, 2].

        num_context_points (int or tuple[int, int], optional): A fixed number of context

            points or a lower and upper bound. Defaults to the range `(1, 50)`.

        num_target_points (int or tuple[int, int], optional): A fixed number of target

            points or a lower and upper bound. Defaults to the fixed number `50`.

        proportion_class (str, optional): Proportion of the data set to split off for

            classification. Defaults to `random`.

        shift (float, optional): Shift between the regression data set and classification

            data set. Defaults to `0.5`.
        
        mode (str, optional): Location of target and context points can either be `random`

            or `disjoint`. Defaults to `random`.

        device (str, optional): Device on which to generate data. If no device is given,

            it will try to use the GPU.

    """



    def __init__(

        self,

        kernel=stheno.EQ().stretch(0.25),

        noise=5e-2,

        seed=0,

        batch_size=16,

        num_tasks=2 ** 14,

        x_range=(-2, 2),

        num_context_points=(2, 50),

        num_target_points=50,

        proportion_class="random",

        shift=0.5,

        mode="random", 

        device=None,

    ):

        self.kernel = kernel

        self.noise = noise



        self.batch_size = batch_size

        self.num_tasks = num_tasks

        self.num_batches = num_tasks // batch_size

        if self.num_batches * batch_size != num_tasks:

            raise ValueError(

                f"Number of tasks {num_tasks} must be a multiple of "

                f"the batch size {batch_size}."

            )

        self.x_range = x_range



        # Ensure that `num_context_points` and `num_target_points` are tuples of lower

        # bounds and upper bounds.

        if not isinstance(num_context_points, tuple):

            num_context_points = (num_context_points, num_context_points)

        if not isinstance(num_target_points, tuple):

            num_target_points = (num_target_points, num_target_points)



        self.num_context_points = num_context_points

        self.num_target_points = num_target_points



        self.proportion_class = proportion_class

        self.shift = shift

        self.mode = mode



        if device is None:

            if torch.cuda.is_available():

                self.device = "cuda"

            else:

                self.device = "cpu"

        else:

            self.device = device



        # The random state must be created on the right device.

        with B.on_device(self.device):

            self.state = B.create_random_state(torch.float32, seed)



    def generate_batch(self):

        """Generate a batch.



        Returns:

            dict: A task, which is a dictionary with keys `x_context`, `y_context`,

                `x_target`, and `y_target`.

        """

        # Sample number of context and target points.

        lower, upper = self.num_context_points

        num_context_points = torch.randint(

            lower, upper + 1, (), generator=self.state, device=self.device

        )

        lower, upper = self.num_target_points

        num_target_points = torch.randint(

            lower, upper + 1, (), generator=self.state, device=self.device

        )



        # Split off in classification and regression numbers.

        if self.proportion_class == "random":

            num_class_context_points = torch.randint(

                1, num_context_points, (), generator=self.state, device=self.device

            )

            num_reg_context_points = num_context_points - num_class_context_points

            num_class_target_points = torch.randint(

                1, num_target_points, (), generator=self.state, device=self.device

            )

            num_reg_target_points = num_target_points - num_class_target_points

        else:

            num_class_context_points = int(self.proportion_class*num_context_points)

            num_reg_context_points = num_context_points - num_class_context_points

            num_class_target_points = int(self.proportion_class*num_target_points)

            num_reg_target_points = num_target_points - num_class_target_points



        def sample_x(num, x_range=self.x_range):

            with B.on_device(self.device):

                lower, upper = x_range

                shape = (self.batch_size, int(num), 1)

                self.state, rand = B.rand(self.state, torch.float32, *shape)

                return lower + rand * (upper - lower)



        # Sample inputs.

        if self.mode == "disjoint":

            x_middle = (self.x_range[1] + self.x_range[0]) / 2 # Question: does this need to be enforced as an integer?

            x_range_left = (self.x_range[0], x_middle)

            x_range_right = (x_middle, self.x_range[1])

            x_context_reg = sample_x(num_reg_context_points, x_range_left)

            x_context_class = sample_x(num_class_context_points, x_range_right)

            x_target_reg = sample_x(num_reg_target_points, x_range_right)

            x_target_class = sample_x(num_class_target_points, x_range_left)

        else:

            x_context_reg = sample_x(num_reg_context_points)

            x_context_class = sample_x(num_class_context_points)

            x_target_reg = sample_x(num_reg_target_points)

            x_target_class = sample_x(num_class_target_points)



        # Sample outputs, taking into account the shift.

        with B.on_device(self.device):

            noise = B.to_active_device(self.noise)

            # Subtract `self.shift` from the classification inputs to induce a

            # shift on the classification data.

            x = B.concat(

                x_context_reg,

                x_context_class - self.shift,

                x_target_reg,

                x_target_class - self.shift,

                axis=1,

            )

            self.state, y = stheno.GP(self.kernel)(x, noise).sample(self.state)



        # Split sampled output.

        i = 0

        splits = []

        for length in [

            num_reg_context_points,

            num_class_context_points,

            num_reg_target_points,

            num_class_target_points,

        ]:

            splits.append(y[:, i : i + length, :])

            i += length



        def to_class(y):

            return (B.sign(y) + 1) / 2
            

        batch = []

        batch.append({

            "type": "classification",

            "x_context": x_context_class,

            "y_context": to_class(splits[1]),

            "x_target": x_target_class,

            "y_target": to_class(splits[3]),

        })

        batch.append({

            "type": "regression",

            "x_context": x_context_reg,

            "y_context": splits[0],

            "x_target": x_target_reg,

            "y_target": splits[2],

        })

        return batch



    def epoch(self):

        """Construct a generator for an epoch.



        Returns:

            generator: Generator for an epoch.

        """



        def lazy_gen_batch():

            return self.generate_batch()



        return (lazy_gen_batch() for _ in range(self.num_batches))

