from client import TabularActorCritic

def main():
    ac = TabularActorCritic(["state_0", "state_1"], ["up", "down"])
    for _ in range(30):
        step = ac.update("state_0", "up", 2.0, "state_1")
    print("Advantage Actor-Critic Verification:")
    print(f"State 0 Value: {ac.V['state_0']:.4f}")
    print(f"Policy: {step['policy_probs']}")

if __name__ == "__main__":
    main()
