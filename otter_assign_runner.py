#!/usr/bin/env python3
"""
Otter Assign Runner Script
Runs otter assign on Jupyter notebooks from raw_notebooks folder.
Generates student and instructor versions in otterized/ and copies autograder zips to autograder_zips/.
Supports both single notebook and batch processing with threading.
"""

import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import logging
import nbformat
from nbclient import NotebookClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OtterAssignRunner:
    def __init__(self, raw_notebooks_path: str, output_dir: str = "otterized", allow_errors: bool = False,
                 wrap_otter_ignore: bool = True, kernel_name: str = "data271", clear_outputs: bool = False,
                 config_path: str = "notebooks_to_otterize.txt"):
        """
        Initialize the Otter Assign Runner.
        
        Args:
            raw_notebooks_path: Path to the raw_notebooks folder
            output_dir: Directory to store otterized notebooks (default: otterized)
            allow_errors: If True, continue notebook execution even when cells raise exceptions
            wrap_otter_ignore: If True, skip cells tagged with 'otter_ignore' during pre-execution
            kernel_name: Name of the Jupyter kernel to use for notebook execution (default: data271)
            clear_outputs: If True, clear cell outputs after otter assign to reduce git noise
            config_path: Path to config file with notebook list (default: notebooks_to_otterize.txt)
        """
        self.raw_notebooks_path = Path(raw_notebooks_path)
        self.output_dir = Path(output_dir)
        self.allow_errors = allow_errors
        self.wrap_otter_ignore = wrap_otter_ignore
        self.kernel_name = kernel_name
        self.clear_outputs_flag = clear_outputs
        self.config_path = Path(config_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.raw_notebooks_path.exists():
            raise FileNotFoundError(f"raw_notebooks path not found: {raw_notebooks_path}")
        
        logger.info(f"Raw notebooks path: {self.raw_notebooks_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Kernel name: {self.kernel_name}")
        if self.allow_errors:
            logger.info("Notebook pre-execution: allow_errors=True (will continue on exceptions)")
        if self.wrap_otter_ignore:
            logger.info("Notebook pre-execution: skipping cells tagged 'otter_ignore'")
        if self.clear_outputs_flag:
            logger.info("Post-processing: will clear cell outputs after otter assign")
        logger.info(f"Config file: {self.config_path}")

    def find_notebooks(self, config_file: str = None) -> List[Path]:
        """Find notebooks to process from the required config file."""
        cfg = Path(config_file) if config_file else self.config_path

        if not cfg.exists():
            raise FileNotFoundError(f"Config file not found: {cfg}")

        logger.info(f"Reading notebooks from config file: {cfg}")
        notebooks = []
        with open(cfg, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    notebook_path = self.raw_notebooks_path.parent / line
                    if notebook_path.exists():
                        notebooks.append(notebook_path)
                    else:
                        logger.warning(f"Notebook not found: {notebook_path}")

        notebooks.sort()
        return notebooks
    
    def get_output_path(self, notebook: Path) -> Path:
        """Get the output path for an otterized notebook."""
        relative_path = notebook.relative_to(self.raw_notebooks_path)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def execute_notebook(self, notebook: Path) -> Tuple[bool, str]:
        """
        Execute all cells in the notebook using nbclient before running otter assign.

        The notebook is executed with its folder as the working directory so that
        relative file paths (e.g., data files) resolve correctly.

        Args:
            notebook: Path to the notebook to execute

        Returns:
            (success, message): True if executed successfully, else False with error message
        """
        try:
            logger.info(f"Running all cells with nbclient: {notebook.name}")
            # Read the notebook
            nb = nbformat.read(str(notebook), as_version=4)

            # Skip cells tagged with 'otter_ignore' by clearing their source
            if self.wrap_otter_ignore:
                for cell in nb.cells:
                    if getattr(cell, 'cell_type', None) == 'code':
                        tags = getattr(cell, 'metadata', {}).get('tags', []) if hasattr(cell, 'metadata') else []
                        if isinstance(tags, list) and 'otter_ignore' in tags:
                            cell.source = ""  # Clear source to skip execution
                            logger.info(f"Skipping 'otter_ignore' tagged cell in {notebook.name}")

            # Execute with notebook directory as path so file IO works
            # Use the kernel specified during initialization (matches environment.yaml)
            # This ensures consistency with otter grade environment
            resources = {"metadata": {"path": str(notebook.parent)}}
            client = NotebookClient(nb, timeout=None, resources=resources, allow_errors=self.allow_errors, kernel_name=self.kernel_name)
            client.execute()

            # Write the executed notebook back to persist outputs for otter assign
            nbformat.write(nb, str(notebook))
            logger.info(f"✓ Executed all cells and saved outputs: {notebook.name}")
            return True, "Executed"
        except Exception as e:
            logger.error(f"✗ Failed executing cells for {notebook.name}: {e}")
            return False, str(e)

    def clear_execution_metadata(self, notebook: Path) -> None:
        """Clear execution metadata (counts and timing) to reduce git noise."""
        try:
            nb = nbformat.read(str(notebook), as_version=4)

            for cell in nb.cells:
                if getattr(cell, 'cell_type', None) == 'code':
                    cell['execution_count'] = None
                if hasattr(cell, 'metadata'):
                    for key in [
                        'execution',
                        'ExecuteTime',
                        'execution_start_time',
                        'execution_end_time',
                    ]:
                        cell.metadata.pop(key, None)

            nbformat.write(nb, str(notebook))
            logger.info(f"✓ Cleared execution metadata: {notebook.name}")
        except Exception as e:
            logger.warning(f"Could not clear execution metadata for {notebook.name}: {e}")

    def clear_outputs(self, notebook: Path) -> None:
        """Clear cell outputs to avoid committing large diffs."""
        try:
            nb = nbformat.read(str(notebook), as_version=4)

            for cell in nb.cells:
                if getattr(cell, 'cell_type', None) == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None

            nbformat.write(nb, str(notebook))
            logger.info(f"✓ Cleared outputs: {notebook.name}")
        except Exception as e:
            logger.warning(f"Could not clear outputs for {notebook.name}: {e}")

    def sync_ids_and_clean_generated(self, source_nb: Path, target_nb: Path) -> None:
        """
        Sync cell IDs from source notebook and clean execution metadata from generated notebooks.
        
        This reduces git churn by ensuring cell IDs remain stable and removing execution artifacts.
        
        Args:
            source_nb: Path to the source (original) notebook
            target_nb: Path to the generated notebook to clean
        """
        if not target_nb.exists():
            return

        try:
            source = nbformat.read(str(source_nb), as_version=4)
            target = nbformat.read(str(target_nb), as_version=4)

            # Sync cell IDs from source to target (matching by index)
            for i, (s_cell, t_cell) in enumerate(zip(source.cells, target.cells)):
                if 'id' in s_cell:
                    t_cell['id'] = s_cell['id']

            for t_cell in target.cells:
                # Clean execution metadata
                if t_cell.get('cell_type') == 'code':
                    t_cell['execution_count'] = None
                    
                # Clean execution-related metadata keys
                if 'metadata' in t_cell:
                    for key in ['execution', 'ExecuteTime', 'execution_start_time', 'execution_end_time']:
                        if key in t_cell['metadata']:
                            del t_cell['metadata'][key]

                # Optionally clear outputs
                if self.clear_outputs_flag and t_cell.get('cell_type') == 'code':
                    t_cell['outputs'] = []

            nbformat.write(target, str(target_nb))
            logger.info(f"✓ Cleaned generated notebook: {target_nb.relative_to(self.output_dir)}")
        except Exception as e:
            logger.warning(f"Could not clean {target_nb.name}: {e}")
    
    def copy_autograder_zip(self, notebook: Path) -> bool:
        """
        Copy the generated autograder.zip from otterized output to autograder_zips folder.
        Renames it from hw01-autograder-[timestamp].zip to hw01-autograder.zip
        
        Args:
            notebook: Path to the notebook
            
        Returns:
            True if copy was successful or file doesn't exist, False on error
        """
        try:
            # Get relative path to determine hw/lab and assignment name
            relative_path = notebook.relative_to(self.raw_notebooks_path)
            parts = relative_path.parts

            # Extract folder type (hw/lab) and assignment name
            if len(parts) < 2:
                return True  # Skip if can't determine structure
            
            folder_type = parts[0]  # 'hw' or 'lab'
            assignment_dir = parts[1]  # 'hw01', 'lab01', etc.
            
            # Look for autograder folder in the output path
            output_path = self.get_output_path(notebook)
            # output_path includes the notebook filename, we need the parent directory
            autograders_dir = output_path.parent / "autograder"

            if not autograders_dir.exists():
                return True  # autograder directory doesn't exist, skip silently
            
            # Find the autograder zip file (hw01-autograder-[timestamp].zip)
            autograder_files = list(autograders_dir.glob(f"{assignment_dir}-autograder_*.zip"))
            if not autograder_files:
                return True  # No autograder zip found, skip silently
            
            # Use the first match
            source_zip = autograder_files[0]
            
            # Create output directory structure: autograder_zips/hw or autograder_zips/lab
            autograder_zips_base = Path.cwd() / "autograder_zips" / folder_type
            autograder_zips_base.mkdir(parents=True, exist_ok=True)
            
            # Create new filename
            new_filename = f"{assignment_dir}-autograder.zip"
            new_zip_path = autograder_zips_base / new_filename
            
            # Copy the file
            shutil.copy2(source_zip, new_zip_path)
            logger.info(f"✓ Copied autograder.zip to: {str(new_zip_path.relative_to(Path.cwd()))}")
            return True
        
        except Exception as e:
            logger.warning(f"Could not copy autograder.zip for {notebook.name}: {str(e)}")
            return True  # Don't fail the entire process for this

    def relocate_outputs(self, notebook: Path, output_path: Path) -> None:
        """
        Reorganize otter assign outputs into student/instructor folders.
        
        Moves student notebooks to otterized/student/{hw,lab}/assignment/
        Moves instructor notebooks and autograder assets to otterized/instructor/{hw,lab}/assignment/
        Autograder contents are flattened (no subfolder), and zip files are excluded.
        
        Args:
            notebook: Path to the source notebook
            output_path: Path to the otter assign output directory
        """
        try:
            relative_parent = notebook.relative_to(self.raw_notebooks_path).parent

            student_src = output_path / "student"
            autograder_src = output_path / "autograder"
            solution_src = output_path / notebook.name

            student_dst = self.output_dir / "student" / relative_parent
            instructor_dst = self.output_dir / "instructor" / relative_parent

            # Ensure destination parents exist
            student_dst.parent.mkdir(parents=True, exist_ok=True)
            instructor_dst.parent.mkdir(parents=True, exist_ok=True)

            # Move student folder
            if student_src.exists():
                if student_dst.exists():
                    shutil.rmtree(student_dst)
                shutil.move(str(student_src), str(student_dst))
                logger.info(f"✓ Moved student files to: {student_dst.relative_to(self.output_dir)}")

            # Move instructor assets: solution + autograder contents (flattened)
            instructor_dst.mkdir(parents=True, exist_ok=True)

            if solution_src.exists():
                dest_solution = instructor_dst / notebook.name
                if dest_solution.exists():
                    dest_solution.unlink()
                shutil.move(str(solution_src), str(dest_solution))

            # Move autograder contents directly into instructor folder (drop zip files)
            if autograder_src.exists():
                for item in autograder_src.iterdir():
                    if item.suffix == '.zip':
                        continue  # Skip zip files (already copied to autograder_zips)
                    dest_item = instructor_dst / item.name
                    if dest_item.exists():
                        if dest_item.is_dir():
                            shutil.rmtree(dest_item)
                        else:
                            dest_item.unlink()
                    shutil.move(str(item), str(dest_item))
                # Remove now-empty autograder folder
                autograder_src.rmdir()
                logger.info(f"✓ Moved instructor files to: {instructor_dst.relative_to(self.output_dir)}")

            # Clean up empty original directory
            if not any(output_path.iterdir()):
                output_path.rmdir()

        except Exception as e:
            logger.warning(f"Could not relocate outputs for {notebook.name}: {e}")
    
    def run_otter_assign(self, notebook: Path) -> Tuple[Path, bool, str]:
        """
        Run complete otter assign workflow on a single notebook.
        
        Workflow:
        1. Execute notebook cells with nbclient
        2. Run otter assign to generate student/instructor versions
        3. Copy autograder zip to autograder_zips/
        4. Clean execution metadata from generated notebooks
        5. Sync cell IDs and optionally clear outputs
        6. Relocate outputs to student/instructor folders
        
        Args:
            notebook: Path to the notebook
            
        Returns:
            Tuple of (notebook_path, success, message)
        """
        output_path = self.get_output_path(notebook).parent
        try:
            logger.info(f"Processing: {notebook.name}")

            # Execute the notebook first to ensure all cells run
            executed, exec_msg = self.execute_notebook(notebook)
            if not executed:
                return (notebook, False, f"nbclient execution failed: {exec_msg}")
            
            # Run otter assign command
            cmd = ["otter", "assign", str(notebook), str(output_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully processed: {notebook.name}")
                
                # Copy autograder.zip if it exists
                self.copy_autograder_zip(notebook)

                # Clear execution metadata to avoid committing noise
                self.clear_execution_metadata(notebook)

                # Optionally clear outputs to further reduce git churn
                if self.clear_outputs_flag:
                    self.clear_outputs(notebook)

                # Sync IDs and clean metadata/outputs in generated notebooks
                generated_targets = [
                    output_path / notebook.name,
                    output_path / "student" / notebook.name,
                    output_path / "autograder" / notebook.name,
                ]
                for target_nb in generated_targets:
                    self.sync_ids_and_clean_generated(notebook, target_nb)

                # Move generated outputs into student/instructor buckets
                self.relocate_outputs(notebook, output_path)
                
                return (notebook, True, "Success")
            else:
                error_msg = result.stderr or result.stdout
                logger.error(f"✗ Failed to process {notebook.name}: {error_msg}")
                return (notebook, False, f"otter assign failed: {error_msg}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout processing: {notebook.name}")
            return (notebook, False, "Process timeout")
        except Exception as e:
            logger.error(f"✗ Error processing {notebook.name}: {str(e)}")
            return (notebook, False, str(e))
    
    def process_single(self, notebook_name: str) -> bool:
        """
        Process a single notebook by name.
        
        Args:
            notebook_name: Name or partial path of the notebook
            
        Returns:
            True if successful, False otherwise
        """
        notebooks = self.find_notebooks()
        
        # Find notebook matching the name
        matching = [nb for nb in notebooks if notebook_name in str(nb)]
        
        if not matching:
            logger.error(f"No notebook found matching: {notebook_name}")
            logger.info("Available notebooks:")
            for nb in notebooks:
                logger.info(f"  - {nb.relative_to(self.raw_notebooks_path.parent)}")
            return False
        
        if len(matching) > 1:
            logger.warning(f"Multiple notebooks match '{notebook_name}':")
            for nb in matching:
                logger.info(f"  - {nb.relative_to(self.raw_notebooks_path.parent)}")
            logger.info(f"Processing first match: {matching[0].name}")
        
        notebook, success, message = self.run_otter_assign(matching[0])
        return success
    
    def process_all(self, num_threads: int = 4, config_file: str = None) -> Tuple[int, int, List[Tuple]]:
        """
        Process notebooks using threading.
        
        Args:
            num_threads: Number of concurrent threads
            config_file: Optional path to config file listing notebooks to process
            
        Returns:
            Tuple of (successful_count, failed_count, failures_list)
        """
        notebooks = self.find_notebooks(config_file)
        logger.info(f"Found {len(notebooks)} notebooks to process")
        
        if not notebooks:
            logger.warning("No notebooks found!")
            return (0, 0, [])
        
        results = []
        failures = []
        
        logger.info(f"Starting processing with {num_threads} threads...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self.run_otter_assign, nb): nb
                for nb in notebooks
            }
            
            for future in as_completed(futures):
                notebook, success, message = future.result()
                results.append((notebook, success, message))
                
                if not success:
                    failures.append((notebook, message))
        
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        logger.info("\n" + ("=" * 60))
        logger.info("Processing complete!")
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Failed: {failed}/{len(results)}")
        logger.info("=" * 60)
        
        if failures:
            logger.warning("\nFailed notebooks:")
            for notebook, message in failures:
                logger.warning(f"  - {notebook.relative_to(self.raw_notebooks_path)}")
                logger.warning(f"    {message}")
        
        return (successful, failed, failures)


def main():
    parser = argparse.ArgumentParser(
        description="Run otter assign on Jupyter notebooks in raw_notebooks folder (default: process all notebooks)"
    )
    
    parser.add_argument(
        "--notebook",
        type=str,
        help="Process a single notebook by name or partial path"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all notebooks using threading"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for batch processing (default: 1; increase carefully)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="otterized",
        help="Output directory for otterized notebooks (default: otterized)"
    )
    parser.add_argument(
        "--raw-notebooks-dir",
        type=str,
        default="raw_notebooks",
        help="Path to raw_notebooks folder (default: raw_notebooks)"
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Continue notebook execution even when cells raise exceptions"
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default="data271",
        help="Jupyter kernel name to use for notebook execution (default: data271 from environment.yaml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="notebooks_to_otterize.txt",
        help="Path to required config file listing notebooks to process (default: notebooks_to_otterize.txt; lines are relative to raw_notebooks parent)"
    )
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Clear code cell outputs after otter assign to minimize git diffs"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path.cwd()
    raw_notebooks_path = (script_dir / args.raw_notebooks_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    try:
        runner = OtterAssignRunner(
            raw_notebooks_path,
            output_dir,
            allow_errors=args.allow_errors,
            kernel_name=args.kernel_name,
            clear_outputs=args.clear_outputs,
            config_path=args.config,
        )
        
        if args.notebook:
            # Process single notebook
            logger.info(f"Processing single notebook: {args.notebook}")
            success = runner.process_single(args.notebook)
            sys.exit(0 if success else 1)
        
        elif args.all:
            # Process all notebooks
            logger.info("Processing all notebooks with threading...")
            successful, failed, failures = runner.process_all(args.threads, args.config)
            sys.exit(0 if failed == 0 else 1)
        
        else:
            # Default: process all notebooks with provided thread count
            logger.info("No arguments provided; processing all notebooks with defaults...")
            successful, failed, failures = runner.process_all(args.threads, args.config)
            sys.exit(0 if failed == 0 else 1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
