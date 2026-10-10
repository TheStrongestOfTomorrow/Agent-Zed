"""Exclusive Job Delegation and Management System for zed-cli."""

import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Callable

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Job:
    """Represents an assigned task to be performed by specific agent personas."""

    def __init__(
        self,
        title: str,
        description: str,
        assigned_agents: List[str],
        priority: JobPriority = JobPriority.MEDIUM,
        job_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.id = job_id or f"JOB-{uuid.uuid4().hex[:6].upper()}"
        self.title = title
        self.description = description
        self.assigned_agents = assigned_agents
        self.priority = priority
        self.status = JobStatus.PENDING
        self.progress = 0.0  # 0.0 to 100.0
        self.status_message = "Job queued"
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.context = context or {}
        self.logs: List[str] = []

    def add_log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "assigned_agents": self.assigned_agents,
            "priority": self.priority.name,
            "status": self.status.value,
            "progress": f"{self.progress:.1f}%",
            "status_message": self.status_message,
            "created_at": self.created_at,
            "logs_count": len(self.logs)
        }

class JobManager:
    """Manages active, pending, and completed workforce jobs for zed-cli."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.job_handlers: Dict[str, Callable] = {}

    def create_job(
        self,
        title: str,
        description: str,
        assigned_agents: List[str],
        priority: JobPriority = JobPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> Job:
        job = Job(title, description, assigned_agents, priority=priority, context=context)
        self.jobs[job.id] = job
        job.add_log(f"Job created by CEO. Assigned to: {', '.join(assigned_agents)}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[JobStatus] = None) -> List[Job]:
        jobs_list = list(self.jobs.values())
        if status_filter:
            jobs_list = [j for j in jobs_list if j.status == status_filter]
        return sorted(jobs_list, key=lambda j: j.priority.value, reverse=True)

    def update_job_progress(self, job_id: str, progress: float, status_message: str):
        job = self.jobs.get(job_id)
        if job:
            job.progress = max(0.0, min(100.0, progress))
            job.status_message = status_message
            job.add_log(f"[{job.progress:.0f}%] {status_message}")

    def mark_completed(self, job_id: str, result: Any, summary: str = "Job finished successfully"):
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.status_message = summary
            job.result = result
            job.completed_at = time.time()
            job.add_log(f"Completed: {summary}")

    def mark_failed(self, job_id: str, error: str):
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error
            job.status_message = f"Failed: {error}"
            job.completed_at = time.time()
            job.add_log(f"Error: {error}")

    def get_summary_report(self) -> str:
        total = len(self.jobs)
        if total == 0:
            return "No active or historical jobs in queue."

        pending = sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)
        running = sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)
        completed = sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED)

        lines = [
            "=== AGENT-ZED WORKFORCE JOB REPORT ===",
            f"Total Jobs: {total} | Running: {running} | Pending: {pending} | Completed: {completed} | Failed: {failed}",
            "--------------------------------------------------------------------------------"
        ]

        for job in self.list_jobs():
            agents_str = ", ".join(job.assigned_agents)
            lines.append(
                f"[{job.id}] {job.title:<30} | Status: {job.status.value:<10} | Progress: {job.progress:>5.1f}% | Agents: {agents_str}"
            )
            if job.status_message:
                lines.append(f"  └─ Status: {job.status_message}")

        return "\n".join(lines)
