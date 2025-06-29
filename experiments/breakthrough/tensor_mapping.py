from core import TensorMapping, OpenSpielState

class BreakthroughTensorMapping(TensorMapping[str, OpenSpielState]):
    @staticmethod
    def encode_states(states: List[OpenSpielState]) -> torch.Tensor:
        return torch.tensor([state.observation_tensor() for state in states])

    @staticmethod
    def decode_states(tensor: torch.Tensor) -> List[OpenSpielState]:
        return [OpenSpielState.from_observation_tensor(tensor[i]) for i in range(tensor.shape[0])]