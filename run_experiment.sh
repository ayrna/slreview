#!/bin/bash

PYTHONEXECUTABLE=/path/to/your/python/executable

if [ "$PYTHONEXECUTABLE" == "/path/to/your/python/executable" ]
then
    echo "Please set the PYTHONEXECUTABLE variable to your Python executable path."
    exit 1
fi

$PYTHONEXECUTABLE ./run_experiment.py $*