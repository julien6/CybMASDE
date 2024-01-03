# Velociraptor - Endpoint visibility and collection tool.

Velociraptor is a tool for collecting host based state information using The Velociraptor Query Language (VQL) queries.

![alt text](/assets/library/nodes/N001/thumbnail.png)

## Supported platforms

Velociraptor is written in Golang and so is available for all the
platforms [supported by Go](https://github.com/golang/go/wiki/MinimumRequirements).
This means that Windows XP and Windows server 2003 are **not**
supported but anything after Windows 7/Vista is.

We build our releases using the MUSL library (x64) for Linux and a
recent MacOS system, so earlier platforms may not be supported by our
release pipeline. We also distribute 32 bit binaries for Windows but
not for Linux. If you need 32 bit Linux builds you will need to build
from source. You can do this easily by forking the project on GitHub,
enabling GitHub Actions in your fork and editing the `Linux Build All
Arches` pipeline.

## Artifact Exchange

Velociraptor's power comes from `VQL Artifacts`, that define many
capabilities to collect many types of data from endpoints.
Velociraptor comes with many built in `Artifacts` for the most common
use cases. The community also maintains a large number of additional
artifacts through the [Artifact Exchange](https://docs.velociraptor.app/exchange/).

## Knowledge Base

If you need help performing a task such as deployment, VQL queries
etc. Your first port of call should be the Velociraptor Knowledge Base
at https://docs.velociraptor.app/knowledge_base/ where you will find
helpful tips and hints.
