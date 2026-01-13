#!/usr/bin/env python3
"""
Otter Grade Runner Script
Runs otter grade on instructor notebooks using autograder zips from autograder_zips folder.
"""

import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OtterGradeRunner:
    def __init__(self, otterized_path: str, verbose: bool = False):
        """
        Initialize the Otter Grade Runner.
        
        Args:
            otterized_path: Path to the otterized folder
            verbose: If True, stream otter grade logs to stdout
        """
        self.otterized_path = Path(otterized_path)
        self.grading_results_dir = Path.cwd() / "grading_results"
        self.autograder_zips_dir = Path.cwd() / "autograder_zips"
        self.verbose = verbose
        
        if not self.otterized_path.exists():
            raise FileNotFoundError(f"otterized path not found: {otterized_path}")
        
        if not self.autograder_zips_dir.exists():
            raise FileNotFoundError(f"autograder_zips folder not found: {self.autograder_zips_dir}")
        
        # Create grading_results directory if it doesn't exist
        self.grading_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Otterized path: {self.otterized_path}")
        logger.info(f"Autograder zips: {self.autograder_zips_dir}")
        logger.info(f"Results directory: {self.grading_results_dir}")
        if self.verbose:
            logger.info("Verbose mode: otter grade logs will be displayed")
    
    def find_autograder_zips(self) -> List[Tuple[Path, Path]]:
        """
        Find all autograder zips and their corresponding instructor notebooks.
        
        Looks for zips in autograder_zips/{hw,lab}/ and notebooks in otterized/instructor/.
        
        Returns:
            List of tuples (notebook_path, autograder_zip_path)
        """
        pairs = []
        instructor_path = self.otterized_path / "instructor"
        
        if not instructor_path.exists():
            logger.warning(f"Instructor folder not found: {instructor_path}")
            return pairs
        
        # Find all autograder zips in autograder_zips folder
        for zip_file in self.autograder_zips_dir.rglob("*.zip"):
            # Extract assignment info from zip filename (e.g., hw01-autograder.zip -> hw01)
            # zip_file.stem removes .zip, then remove -autograder suffix
            assignment_name = zip_file.stem.replace("-autograder", "")
            
            # Determine folder type (hw or lab) from parent directory
            folder_type = zip_file.parent.name  # 'hw' or 'lab'
            
            # Look for corresponding instructor notebook
            notebook_path = instructor_path / folder_type / assignment_name / f"{assignment_name}.ipynb"
            
            if notebook_path.exists():
                pairs.append((notebook_path, zip_file))
            else:
                logger.warning(f"No instructor notebook found for {zip_file.name} at {notebook_path}")
        
        return pairs
    
    def run_otter_grade(self, notebook: Path, autograder_zip: Path) -> Tuple[Path, bool, str]:
        """
        Run otter grade on an instructor notebook using its autograder zip.
        
        Args:
            notebook: Path to the instructor notebook
            autograder_zip: Path to the autograder zip
            
        Returns:
            Tuple of (notebook_path, success, message)
        """
        try:
            logger.info(f"Grading: {notebook.name}")
            
            # Get assignment name from notebook (e.g., hw01 from hw01.ipynb)
            assignment_name = notebook.stem
            
            # Run otter grade command
            cmd = ["otter", "grade", "-a", str(autograder_zip), str(notebook), "-n", assignment_name]
            
            if self.verbose:
                # Stream output directly to stdout
                result = subprocess.run(
                    cmd,
                    cwd=notebook.parent
                )
            else:
                # Capture output silently
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=notebook.parent
                )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully graded: {notebook.name}")
                
                # Copy final_grades.csv if it exists
                self.copy_grade_results(notebook.parent, assignment_name)
                
                return (notebook, True, "Success")
            else:
                if self.verbose:
                    # Output was already shown; just log the error
                    logger.error(f"✗ Failed to grade {notebook.name}")
                    return (notebook, False, "otter grade failed")
                else:
                    error_msg = result.stderr or result.stdout
                    logger.error(f"✗ Failed to grade {notebook.name}: {error_msg}")
                    return (notebook, False, f"otter grade failed: {error_msg}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout grading: {notebook.name}")
            return (notebook, False, "Process timeout")
        except Exception as e:
            logger.error(f"✗ Error grading {notebook.name}: {str(e)}")
            return (notebook, False, str(e))
    
    def copy_grade_results(self, work_dir: Path, assignment_name: str) -> bool:
        """
        Move and rename final_grades.csv into grading_results to avoid leaving temp files.
        
        Args:
            work_dir: Directory where otter grade was run
            assignment_name: Name of the assignment (e.g., 'hw01')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_csv = work_dir / "final_grades.csv"
            
            if not source_csv.exists():
                logger.warning(f"final_grades.csv not found at {source_csv}")
                return False
            
            # Destination filename: hw01_grading_results.csv
            dest_filename = f"{assignment_name}_grading_results.csv"
            dest_csv = self.grading_results_dir / dest_filename

            # Remove existing dest to avoid cross-filesystem rename issues
            if dest_csv.exists():
                dest_csv.unlink()
            
            # Move (rename) to keep autograder folders clean
            shutil.move(str(source_csv), str(dest_csv))
            logger.info(f"✓ Moved results to: {dest_csv.relative_to(Path.cwd())}")
            return True
        
        except Exception as e:
            logger.warning(f"Could not move grade results: {str(e)}")
            return False
    
    def process_single(self, notebook_name: str) -> bool:
        """
        Process a single notebook by name.
        
        Args:
            notebook_name: Name or partial path of the notebook
            
        Returns:
            True if successful, False otherwise
        """
        pairs = self.find_autograder_zips()
        
        # Find notebook matching the name
        matching = [(nb, zp) for nb, zp in pairs if notebook_name in str(nb)]
        
        if not matching:
            logger.error(f"No notebook found matching: {notebook_name}")
            logger.info("Available notebooks:")
            for nb, _ in pairs:
                logger.info(f"  - {nb.relative_to(self.otterized_path)}")
            return False
        
        if len(matching) > 1:
            logger.warning(f"Multiple notebooks match '{notebook_name}':")
            for nb, _ in matching:
                logger.info(f"  - {nb.relative_to(self.otterized_path)}")
            logger.info(f"Processing first match: {matching[0][0].name}")
        
        notebook, autograder_zip = matching[0]
        notebook_path, success, message = self.run_otter_grade(notebook, autograder_zip)
        return success
    
    def process_all(self, num_threads: int = 4) -> Tuple[int, int, List[Tuple]]:
        """
        Process all notebooks using threading.
        
        Args:
            num_threads: Number of concurrent threads
            
        Returns:
            Tuple of (successful_count, failed_count, failures_list)
        """
        pairs = self.find_autograder_zips()
        logger.info(f"Found {len(pairs)} notebooks to grade")
        
        if not pairs:
            logger.warning("No notebooks found!")
            return (0, 0, [])
        
        results = []
        failures = []
        
        logger.info(f"Starting grading with {num_threads} threads...")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self.run_otter_grade, nb, zp): nb 
                for nb, zp in pairs
            }
            
            for future in as_completed(futures):
                notebook, success, message = future.result()
                results.append((notebook, success, message))
                
                if not success:
                    failures.append((notebook, message))
        
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Grading complete!")
        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Failed: {failed}/{len(results)}")
        logger.info(f"{'='*60}")
        
        if failures:
            logger.warning("\nFailed notebooks:")
            for notebook, message in failures:
                logger.warning(f"  - {notebook.relative_to(self.otterized_path)}")
                logger.warning(f"    {message}")
        
        return (successful, failed, failures)


def main():
    parser = argparse.ArgumentParser(
        description="Run otter grade on instructor notebooks from otterized/instructor folder"
    )
    
    parser.add_argument(
        "--notebook",
        type=str,
        help="Grade a single notebook by name or partial path"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Grade all notebooks using threading"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for batch processing (default: 1; Docker can be resource-intensive)"
    )
    parser.add_argument(
        "--otterized-dir",
        type=str,
        default="otterized",
        help="Path to otterized folder (default: otterized)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream otter grade logs to stdout as notebooks are graded"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path.cwd()
    otterized_path = (script_dir / args.otterized_dir).resolve()
    
    try:
        runner = OtterGradeRunner(otterized_path, verbose=args.verbose)
        
        if args.notebook:
            # Grade single notebook
            logger.info(f"Grading single notebook: {args.notebook}")
            success = runner.process_single(args.notebook)
            sys.exit(0 if success else 1)
        
        elif args.all:
            # Grade all notebooks
            logger.info("Grading all notebooks with threading...")
            successful, failed, failures = runner.process_all(args.threads)
            sys.exit(0 if failed == 0 else 1)
        
        else:
            # Default: show usage
            parser.print_help()
            logger.info("\nExamples:")
            logger.info("  # Grade all notebooks with 4 threads:")
            logger.info("  python otter_grade_runner.py --all")
            logger.info("\n  # Grade all notebooks with 8 threads:")
            logger.info("  python otter_grade_runner.py --all --threads 8")
            logger.info("\n  # Grade with verbose output:")
            logger.info("  python otter_grade_runner.py --notebook hw01.ipynb --verbose")
            logger.info("\n  # Grade single notebook:")
            logger.info("  python otter_grade_runner.py --notebook hw01.ipynb")
            logger.info("\n  # List available notebooks:"))
            logger.info("  python otter_grade_runner.py --notebook nonexistent")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
