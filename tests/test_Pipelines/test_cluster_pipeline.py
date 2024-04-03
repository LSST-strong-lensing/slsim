import pytest
from slsim.Pipelines.cluster_pipeline import ClusterPipeline


class TestClusterPipeline(object):
    def setup_method(self):
        self.pipeline = ClusterPipeline()

    def test_cluster_pipeline_instance(self):
        pipeline = ClusterPipeline()


if __name__ == "__main__":
    pytest.main()
