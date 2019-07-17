from genericWrapper4AC.generic_wrapper import AbstractWrapper
from genericWrapper4AC.domain_specific.satwrapper import SatWrapper
import argparse


class Satenstein_Wrapper(SatWrapper):

    def __init__(self):
        SatWrapper.__init__(self)

    def get_command_line_args(self, runargs, config):
        '''
        @contact:    lindauer@informatik.uni-freiburg.de, fh@informatik.uni-freiburg.de
        Returns the command line call string to execute the target algorithm (here: Satenstein).
        Args:
            runargs: a map of several optional arguments for the execution of the target algorithm.
                    {
                      "instance": <instance>,
                      "specifics" : <extra data associated with the instance>,
                      "cutoff" : <runtime cutoff>,                               !!! should be mapped to timeout !!!
                      "runlength" : <runlength cutoff>,                          !!! should be mapped to cutoff  !!!
                      "seed" : <seed>
                    }
            config: a mapping from parameter name to parameter value
        Returns:
            A command call list to execute the target algorithm.
        '''
        solver_binary = "satenstein/ubcsat"
        print('##################################################')

        # Construct the call string to glucose.
        cmd = "%s -alg satenstein" % (solver_binary)
        

        for name, value in config.items():
            #TODO
            cmd += " " + name + " " + value

        #TODO rest of the command
        cmd += " -seed " + str(runargs["seed"])
        cmd += " -inst " + runargs["instance"]
        cmd += " -target 0 -r satcomp"
        if runargs["runlength"] == 0:
            cmd += " -cutoff -1"
        else:           
            cmd += " -cutoff " + str(runargs["runlength"])
        cmd += " -timeout " + str(runargs["cutoff"])

        # remember instance and cmd to verify the result later on
        self._instance = runargs["instance"]
        self._cmd = cmd

        return cmd

if __name__ == "__main__":
    '''
    cmdline_parser = argparse.ArgumentParser('Sat Wrapper command line arguments')
    cmdline_parser.add_argument('-cutoff',
                                default=45,
                                type=float)
    
    args, unknowns = cmdline_parser.parse_known_args()
    print(args.cutoff)
    '''
    wrapper = Satenstein_Wrapper()
    wrapper.main()
