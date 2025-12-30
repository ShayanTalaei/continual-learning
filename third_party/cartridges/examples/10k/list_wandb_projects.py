import wandb
import argparse

api = wandb.Api()

def explore_cartridges_project(entity="cdingg", project="cartridges", max_runs=50):
    """Explore the cartridges project, showing runs and their files."""
    
    project_path = f"{entity}/{project}"
    print(f"Exploring project: {project_path}\n")
    print("="*80)
    
    try:
        # Get all runs
        print(f"Fetching runs (showing up to {max_runs})...")
        runs = api.runs(path=project_path, per_page=max_runs)
        run_list = list(runs)
        
        if not run_list:
            print("  No runs found in this project.")
            return
        
        print(f"Found {len(run_list)} run(s)\n")
        
        # Filter for cartridge-related runs if needed
        cartridge_runs = [r for r in run_list if "cartridge" in r.name.lower() or "cartridge" in str(r.config).lower()]
        
        if cartridge_runs:
            print(f"Found {len(cartridge_runs)} cartridge-related run(s):\n")
            runs_to_show = cartridge_runs
        else:
            print("Showing all runs:\n")
            runs_to_show = run_list
        
        # Show details for each run
        for i, run in enumerate(runs_to_show, 1):
            print(f"{'='*80}")
            print(f"Run {i}/{len(runs_to_show)}: {run.name}")
            print(f"  ID: {run.id}")
            print(f"  State: {run.state}")
            print(f"  Created: {run.created_at}")
            # print(f"  Updated: {run.updated_at}")
            
            # Show config if available
            if run.config:
                print(f"  Config keys: {list(run.config.keys())[:5]}...")  # Show first 5 keys
            
            # List files in this run
            print(f"  Files:")
            try:
                files = list(run.files())
                if not files:
                    print("    (No files)")
                else:
                    # Group files by type
                    cache_files = [f for f in files if "cache" in f.name.lower() or f.name.endswith(".pt")]
                    other_files = [f for f in files if f not in cache_files]
                    
                    if cache_files:
                        print(f"    Cache/Model files ({len(cache_files)}):")
                        for f in cache_files[:10]:  # Show first 10
                            size_mb = f.size / (1024*1024) if f.size else 0
                            print(f"      - {f.name} ({size_mb:.2f} MB)")
                        if len(cache_files) > 10:
                            print(f"      ... and {len(cache_files) - 10} more")
                    
                    if other_files:
                        print(f"    Other files ({len(other_files)}):")
                        for f in other_files[:5]:  # Show first 5
                            size_mb = f.size / (1024*1024) if f.size else 0
                            print(f"      - {f.name} ({size_mb:.2f} MB)")
                        if len(other_files) > 5:
                            print(f"      ... and {len(other_files) - 5} more")
                    
                    print(f"    Total files: {len(files)}")
            except Exception as e:
                print(f"    Error listing files: {e}")
            
            print()
        
        # Summary
        print("="*80)
        print(f"Summary: {len(runs_to_show)} run(s) shown")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Explore WandB cartridges project")
    parser.add_argument("--entity", default="cdingg", help="WandB entity name")
    parser.add_argument("--project", default="cartridges", help="Project name")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum runs to fetch")
    parser.add_argument("--run-name", help="Filter by specific run name (partial match)")
    args = parser.parse_args()
    
    if args.run_name:
        # Search for specific run
        project_path = f"{args.entity}/{args.project}"
        print(f"Searching for runs matching '{args.run_name}' in {project_path}...\n")
        try:
            runs = api.runs(path=project_path, filters={"display_name": {"$regex": args.run_name}})
            run_list = list(runs)
            if run_list:
                print(f"Found {len(run_list)} matching run(s):\n")
                for run in run_list:
                    print(f"  - {run.name} (ID: {run.id})")
                    files = list(run.files())
                    cache_files = [f for f in files if "cache" in f.name.lower() or f.name.endswith(".pt")]
                    print(f"    Cache files: {len(cache_files)}")
                    for f in cache_files[:5]:
                        print(f"      - {f.name}")
            else:
                print("No matching runs found.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        explore_cartridges_project(args.entity, args.project, args.max_runs)

if __name__ == "__main__":
    main()

