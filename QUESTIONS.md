# Questions

## Langgraph

- [ ] why is Messages defined as a union here

```python
Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]
# https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/message.py
```

- [ ] How to use `input` and `retry` function arguments when calling [`add_node`](https://github.com/langchain-ai/langgraph/blob/a93775413281df9ddf6ba29cc388b2460d94b9af/libs/langgraph/langgraph/graph/state.py#L244) on `uncompiled_graph` ?