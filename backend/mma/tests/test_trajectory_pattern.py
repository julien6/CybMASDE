import unittest
from mma_wrapper.trajectory_pattern import trajectory_pattern


class test_trajectory_pattern(unittest.TestCase):

    def test_global(self):
        traj_pattern = trajectory_pattern(
            "0,[1,70,80,2](0,3),3,[#any](0,*),4,5,6")

        trajectory = "0,1,70,80,2,3,10,20,40,4,5"

        match, matched, coverage, next_seq = traj_pattern.match(trajectory)

        print("match: ", match)
        print("matched: ", matched)
        print("coverage: ", coverage)
        print("next_seq: ", next_seq)

        print("trajectory sample: ", traj_pattern.sample())

        print("trajectory_pattern.to_dict(): ", traj_pattern.to_dict())

        print("trajectory: ", traj_pattern)

        # hp = trajectory_patterns()
        # hp.add_pattern("[0,1,2,3,[#any](0,*),4,5,6](1,1)")
        # hp.add_pattern("[0,1,2,7,9](1,1)")
        # # print(hp.get_actions("0,1,2,3,89,10,4", "5"))
        # print(hp.get_actions("0,1", "2"))

        # print(_match("[[0,1](1,1),[[2,3,4](1,2),6,7,8](1,1)](1,1)", "0"))
        # _match2("[0,[1,2](1,1),[[[4,6](0,1),9,7](1,1),8,2,2,2,4](2,2)](1,1)", "2")

        # trajectory_patt = trajectory_pattern(
        #     "[0,[1,2](1,1),3](1,1)")
        # print(trajectory_patt.match("0,1", "2"))


if __name__ == '__main__':
    unittest.main()
