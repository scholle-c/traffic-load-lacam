import argparse
from pathlib import Path

from pycam import (
    LaCAM,
    TrafficLoadConfig,
    get_grid,
    get_scenario,
    save_configs_for_visualizer,
    validate_mapf_solution,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "tunnel.map",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        default=Path(__file__).parent / "assets" / "tunnel.scen",
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--time_limit_ms", type=int, default=1000)
    parser.add_argument(
        "--flg_star",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="choose LaCAM* (default) or vanilla LaCAM",
    )

    # traffic load guidance
    parser.add_argument(
        "--load-mode",
        choices=["off", "tiebreak", "penalty"],
        default="off",
        help="off: vanilla PIBT; tiebreak: (dist, load); penalty: dist + lambda*load",
    )
    parser.add_argument("--load-lambda", type=float, default=1.0)
    parser.add_argument(
        "--load-window",
        type=int,
        default=10,
        help="future steps the load looks at, -1 = whole shortest-path DAG",
    )
    parser.add_argument(
        "--load-update",
        choices=["start", "step", "agent"],
        default="agent",
        help="start: once; step: per configuration; agent: also after each agent",
    )

    args = parser.parse_args()
    load_cfg = TrafficLoadConfig(
        mode=args.load_mode,
        lam=args.load_lambda,
        window=None if args.load_window < 0 else args.load_window,
        update=args.load_update,
    )

    # define problem instance
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    # solve MAPF
    planner = LaCAM()
    solution = planner.solve(
        grid=grid,
        starts=starts,
        goals=goals,
        seed=args.seed,
        time_limit_ms=args.time_limit_ms,
        flg_star=args.flg_star,
        verbose=args.verbose,
        load_cfg=load_cfg,
    )
    validate_mapf_solution(grid, starts, goals, solution)

    # save result
    save_configs_for_visualizer(solution, args.output_file)
