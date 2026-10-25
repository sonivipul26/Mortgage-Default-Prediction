import logging
import os
import sys
from pathlib import Path
import pyspark
from pyspark.sql import SparkSession
from .config import ROOT

def get_spark(app_name="MortgageDefaultPrediction"):
    logging.getLogger(__name__).info("Starting Spark session")
    # On Windows Spark otherwise falls back to a non-existent `python3` worker.
    # Use the active virtual-environment interpreter for both driver and workers.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Prefer the PySpark 3.5.1 distribution pinned in requirements over a
    # machine-wide SPARK_HOME (which may be a different Spark/Scala build).
    os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).resolve().parent)
    # Use a project-local disk/shuffle directory. The OS %TEMP% on Windows can be
    # cleaned concurrently (or scanned by antivirus), which orphaned Spark shuffle
    # files and aborted the class-weighted model fit in earlier runs (the
    # hs_err_pid*.log / replay_pid*.log files in the repo root are the residual
    # JVM out-of-memory crashes from those failed runs).
    local_dir = ROOT / ".spark_tmp"
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_LOCAL_DIRS"] = str(local_dir)
    return (SparkSession.builder.appName(app_name).master("local[2]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.local.dir", str(local_dir))
            .config("spark.driver.memory", "2g")
            .config("spark.driver.maxResultSize", "1g")
            .config("spark.pyspark.python", sys.executable).getOrCreate())
