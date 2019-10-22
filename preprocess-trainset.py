import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField


# reading train set & converting `step` to int
traindata = spark.read.csv("./data/train.csv", header=True).withColumn("step", F.col("step").cast(IntegerType()))

# zipping train data with artificial index 
trainZippedWithArtId = traindata.select("session_id", "step").rdd.map(tuple).sortBy(lambda x: (x[0], x[1])).zipWithIndex().map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["session_id", "step", "idx"])

# finding index around which split will be performed
splitIdx = int(.8 * trainZippedWithArtId.count())
# finding session where such an index is present
splitSession = trainZippedWithArtId.filter(F.col("idx") == splitIdx).select("session_id").first()[0]

# boolean attributes will be used to find splitting point
trainWithBoolAttribs = trainZippedWithArtId.withColumn("ifidx", F.col("idx") < splitIdx).withColumn("ifsess", F.col("session_id") != splitSession)
# udf for finding splitting point
splitUDF = F.udf(lambda idx, sess: idx and sess)
# abstract splitting with boolean
trainSplitFilter = trainWithBoolAttribs.withColumn("split_filter", splitUDF(F.col("ifidx"), F.col("ifsess"))).select("session_id", "step", "split_filter")

# joining train set with the one containing splitting information
joined = traindata.join(trainSplitFilter, ["session_id", "step"], how="left")

# using a mask to split data into two separate DataFrames
mask = joined["split_filter"] == True
mytrain = joined[mask].drop(F.col("split_filter"))
myPartialTest = joined[~mask].drop(F.col("split_filter"))

# aggregating `steps` and `action_types` into lists per session
testTmp = myPartialTest.groupBy("session_id").agg(F.collect_list(F.col("step")).alias("list_steps"), F.collect_list(F.col("action_type")).alias("list_actions"))
# lel = traindf.groupBy("session_id").agg(F.collect_list(F.col("step")).alias("list_step"), F.collect_list(F.col("action_type")).alias("list_action"))

def last_click(steps, actions):
  x = [(s,a) for s, a in zip(steps, actions) if a.startswith("click")]
  return x[-1] if len(x) > 0 else None


# schema for the returned udf required since python is dynamically typed
schema = StructType([StructField("step", IntegerType(), nullable=True), StructField("action", StringType(), nullable=True)])
# udf for finding last clicks per session if present at all
clickudf = F.udf(last_click, schema)

# adding info about last click with use of udf
lastClickInSession = testTmp.withColumn("last_click", clickudf(F.col("list_steps"), F.col("list_actions"))).select("session_id", "last_click.step", "last_click.action").toDF("session_id", "newstep", "newaction")

# udf to change reference into null if statement satisfied
statementudf = F.udf(lambda s, ns, a, na, r: "null" if s == ns and a == na else r)
# final join and replacement of columns in order to get proper format of the test set
mytest = myPartialTest.join(lastClickInSession, ["session_id"], "left").withColumn("reference", statementudf(F.col("step"), F.col("newstep"), F.col("action_type"), F.col("newaction"), F.col("reference"))).drop("newstep")

# testing
reftest = mytest.groupBy("reference").count().orderBy(F.col("count").desc()).first()[1] / mytest.count() * 100.0
print(f"Null references take up to {reftest} % of all datapoints in the test set")

actualtestdata = spark.read.csv("./data/test.csv", header=True)
actualreftest = actualtestdata.groupBy("reference").count().orderBy(F.col("count").desc()).first()[1] / actualtestdata.count() * 100.0
print(f"Whereas null references in the actual test dataset's take up to {actualreftest} % of the datapoints")

assert abs(reftest - actualreftest) < 2.0
