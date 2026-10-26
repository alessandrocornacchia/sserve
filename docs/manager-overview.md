# Inference on demand, without managing GPU jobs

Researchers should ask for a model, not manage a cluster deployment. This service
gives benchmark runs a stable inference address and takes responsibility for
finding and maintaining the compute behind it.

## How people use it

A researcher requests a model, waits for availability, and runs their existing
benchmark. They can do this from any authorized machine. Several researchers can
share the same model service without coordinating job ownership. Capacity is
released when their work finishes or becomes inactive.

## How responsibilities are divided

- **Researchers own their experiments.** They choose the model and run their
  benchmarks; they do not submit or repair serving jobs.
- **The manager owns availability.** It tracks demand, seeks approved compute
  options, repairs lost connections and replaces interrupted capacity.
- **The router owns request distribution.** It directs work to available model
  servers while keeping the address used by researchers unchanged.
- **Model servers own inference.** They load the model, answer requests and
  report whether they can accept more work.
- **Cluster administrators retain control of resources.** Their access,
  scheduling and utilization policies remain authoritative.

## Why this design

Separating the address from the compute lets us use spare capacity without
making every researcher handle its interruptions. Tracking real demand lets us
share resources, add capacity when justified and release it when unused.

The boundary is important: a stable address is not a guarantee of uninterrupted
service. Spare capacity can disappear, and a request already in progress may
fail. The service should make those interruptions visible and recover when
policy and capacity allow—not hide them or bypass cluster rules.
