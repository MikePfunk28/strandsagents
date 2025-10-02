"""Secure reporting system for central audit and monitoring.

Provides encrypted reporting back to the central orchestrator with
comprehensive audit trails and anomaly detection.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of security reports."""
    AGENT_STATUS = "agent_status"
    TASK_COMPLETION = "task_completion"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_METRIC = "performance_metric"
    ANOMALY_DETECTION = "anomaly_detection"
    AUDIT_EVENT = "audit_event"

class Severity(Enum):
    """Severity levels for reports."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityReport:
    """Secure report with encryption and integrity."""
    report_id: str
    report_type: ReportType
    severity: Severity
    agent_id: str
    timestamp: datetime
    data: Dict[str, Any]
    encrypted_data: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    anomaly_id: str
    agent_id: str
    anomaly_type: str
    confidence: float
    description: str
    timestamp: datetime
    evidence: Dict[str, Any]

class SecureReporter:
    """Handles secure reporting and audit trails."""

    def __init__(self, orchestrator_id: str):
        """Initialize secure reporter.

        Args:
            orchestrator_id: ID of the central orchestrator
        """
        self.orchestrator_id = orchestrator_id
        self.reports_sent = 0
        self.reports_failed = 0
        self.audit_trail: List[SecurityReport] = []
        self.anomaly_alerts: List[AnomalyAlert] = []

        # Anomaly detection baselines
        self.agent_baselines: Dict[str, Dict[str, float]] = {}
        self.performance_baselines: Dict[str, Dict[str, float]] = {}

        # Alert thresholds
        self.anomaly_threshold = 0.7
        self.performance_degradation_threshold = 0.3

    async def send_secure_report(self, report: SecurityReport) -> bool:
        """Send a secure report to the orchestrator.

        Args:
            report: Security report to send

        Returns:
            True if report sent successfully
        """
        try:
            # Encrypt sensitive data
            if report.data and any(key in str(report.data).lower() for key in ['password', 'key', 'token', 'secret']):
                report.encrypted_data = await self._encrypt_data(report.data)
                report.data = {"encrypted": True, "size": len(str(report.data))}

            # Add signature
            report.signature = await self._sign_report(report)

            # Store in audit trail
            self.audit_trail.append(report)

            # Simulate sending to orchestrator (would use real MCP in production)
            await self._transmit_report(report)

            self.reports_sent += 1
            logger.info(f"Sent secure report {report.report_id} to orchestrator")

            # Cleanup old audit trail
            await self._cleanup_audit_trail()

            return True

        except Exception as e:
            self.reports_failed += 1
            logger.error(f"Failed to send secure report {report.report_id}: {e}")
            return False

    async def report_agent_status(self, agent_id: str, status_data: Dict[str, Any],
                                severity: Severity = Severity.LOW) -> bool:
        """Report agent status update.

        Args:
            agent_id: Agent identifier
            status_data: Status information
            severity: Report severity

        Returns:
            True if report sent successfully
        """
        report = SecurityReport(
            report_id=f"status_{agent_id}_{int(datetime.now().timestamp())}",
            report_type=ReportType.AGENT_STATUS,
            severity=severity,
            agent_id=agent_id,
            timestamp=datetime.now(),
            data=status_data,
            metadata={
                "source": "agent_monitor",
                "category": "status_update"
            }
        )

        # Check for anomalies in status
        await self._check_status_anomalies(agent_id, status_data)

        return await self.send_secure_report(report)

    async def report_task_completion(self, agent_id: str, task_data: Dict[str, Any],
                                   performance_metrics: Dict[str, float]) -> bool:
        """Report task completion with performance metrics.

        Args:
            agent_id: Agent identifier
            task_data: Task completion data
            performance_metrics: Performance metrics

        Returns:
            True if report sent successfully
        """
        report = SecurityReport(
            report_id=f"task_{task_data.get('task_id', 'unknown')}_{int(datetime.now().timestamp())}",
            report_type=ReportType.TASK_COMPLETION,
            severity=Severity.LOW,
            agent_id=agent_id,
            timestamp=datetime.now(),
            data={
                "task": task_data,
                "performance": performance_metrics
            },
            metadata={
                "source": "task_monitor",
                "category": "completion"
            }
        )

        # Check for performance anomalies
        await self._check_performance_anomalies(agent_id, performance_metrics)

        return await self.send_secure_report(report)

    async def report_security_alert(self, agent_id: str, alert_type: str,
                                  alert_data: Dict[str, Any], severity: Severity) -> bool:
        """Report security alert.

        Args:
            agent_id: Agent identifier
            alert_type: Type of security alert
            alert_data: Alert details
            severity: Alert severity

        Returns:
            True if report sent successfully
        """
        report = SecurityReport(
            report_id=f"alert_{alert_type}_{int(datetime.now().timestamp())}",
            report_type=ReportType.SECURITY_ALERT,
            severity=severity,
            agent_id=agent_id,
            timestamp=datetime.now(),
            data={
                "alert_type": alert_type,
                "details": alert_data
            },
            metadata={
                "source": "security_monitor",
                "category": "alert",
                "alert_type": alert_type
            }
        )

        return await self.send_secure_report(report)

    async def report_anomaly(self, anomaly: AnomalyAlert) -> bool:
        """Report detected anomaly.

        Args:
            anomaly: Anomaly alert

        Returns:
            True if report sent successfully
        """
        # Store anomaly
        self.anomaly_alerts.append(anomaly)

        # Create security report
        severity = Severity.HIGH if anomaly.confidence > 0.8 else Severity.MEDIUM

        report = SecurityReport(
            report_id=f"anomaly_{anomaly.anomaly_id}",
            report_type=ReportType.ANOMALY_DETECTION,
            severity=severity,
            agent_id=anomaly.agent_id,
            timestamp=anomaly.timestamp,
            data={
                "anomaly_type": anomaly.anomaly_type,
                "confidence": anomaly.confidence,
                "description": anomaly.description,
                "evidence": anomaly.evidence
            },
            metadata={
                "source": "anomaly_detector",
                "category": "anomaly"
            }
        )

        return await self.send_secure_report(report)

    async def _check_status_anomalies(self, agent_id: str, status_data: Dict[str, Any]):
        """Check for anomalies in agent status."""
        try:
            # Initialize baseline if not exists
            if agent_id not in self.agent_baselines:
                self.agent_baselines[agent_id] = {}

            current_time = datetime.now()

            # Check for suspicious patterns
            suspicious_indicators = []

            # Check for rapid status changes
            if "status_changes" in status_data:
                changes = status_data["status_changes"]
                if isinstance(changes, int) and changes > 10:  # More than 10 changes recently
                    suspicious_indicators.append(f"Rapid status changes: {changes}")

            # Check for unusual task queue sizes
            if "tasks_queued" in status_data:
                queue_size = status_data["tasks_queued"]
                if isinstance(queue_size, int) and queue_size > 100:
                    suspicious_indicators.append(f"Large task queue: {queue_size}")

            # Check for memory usage anomalies
            if "memory_usage" in status_data:
                memory_usage = status_data["memory_usage"]
                if isinstance(memory_usage, (int, float)) and memory_usage > 0.9:
                    suspicious_indicators.append(f"High memory usage: {memory_usage}")

            # Create anomaly if suspicious indicators found
            if suspicious_indicators:
                anomaly = AnomalyAlert(
                    anomaly_id=f"status_anomaly_{agent_id}_{int(current_time.timestamp())}",
                    agent_id=agent_id,
                    anomaly_type="status_anomaly",
                    confidence=min(0.8, len(suspicious_indicators) * 0.3),
                    description=f"Suspicious status patterns detected: {', '.join(suspicious_indicators)}",
                    timestamp=current_time,
                    evidence={
                        "indicators": suspicious_indicators,
                        "status_data": status_data
                    }
                )

                await self.report_anomaly(anomaly)

        except Exception as e:
            logger.error(f"Error checking status anomalies for {agent_id}: {e}")

    async def _check_performance_anomalies(self, agent_id: str, metrics: Dict[str, float]):
        """Check for performance anomalies."""
        try:
            # Initialize baseline if not exists
            if agent_id not in self.performance_baselines:
                self.performance_baselines[agent_id] = {}

            current_time = datetime.now()
            anomalies_detected = []

            for metric_name, current_value in metrics.items():
                if not isinstance(current_value, (int, float)):
                    continue

                # Update baseline (exponential moving average)
                if metric_name in self.performance_baselines[agent_id]:
                    baseline = self.performance_baselines[agent_id][metric_name]
                    new_baseline = (baseline * 0.9) + (current_value * 0.1)
                    self.performance_baselines[agent_id][metric_name] = new_baseline

                    # Check for significant deviation
                    if baseline > 0:
                        deviation = abs(current_value - baseline) / baseline
                        if deviation > self.performance_degradation_threshold:
                            anomalies_detected.append({
                                "metric": metric_name,
                                "current": current_value,
                                "baseline": baseline,
                                "deviation": deviation
                            })
                else:
                    # First measurement - establish baseline
                    self.performance_baselines[agent_id][metric_name] = current_value

            # Report anomalies if found
            if anomalies_detected:
                confidence = min(0.9, len(anomalies_detected) * 0.2 + 0.5)

                anomaly = AnomalyAlert(
                    anomaly_id=f"perf_anomaly_{agent_id}_{int(current_time.timestamp())}",
                    agent_id=agent_id,
                    anomaly_type="performance_anomaly",
                    confidence=confidence,
                    description=f"Performance anomalies detected in {len(anomalies_detected)} metrics",
                    timestamp=current_time,
                    evidence={
                        "anomalies": anomalies_detected,
                        "full_metrics": metrics
                    }
                )

                await self.report_anomaly(anomaly)

        except Exception as e:
            logger.error(f"Error checking performance anomalies for {agent_id}: {e}")

    async def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data (simplified implementation)."""
        # In production, use proper encryption like AES
        import base64
        import json

        json_data = json.dumps(data)
        encoded_data = base64.b64encode(json_data.encode()).decode()
        return f"ENCRYPTED:{encoded_data}"

    async def _sign_report(self, report: SecurityReport) -> str:
        """Create cryptographic signature for report."""
        import hashlib
        import hmac

        # Create signature of report data
        report_dict = asdict(report)
        report_dict.pop('signature', None)  # Remove signature field
        report_dict.pop('encrypted_data', None)  # Don't sign encrypted data

        # Convert enums to strings for JSON serialization
        for key, value in report_dict.items():
            if hasattr(value, 'value'):  # Handle enum types
                report_dict[key] = value.value

        canonical_data = json.dumps(report_dict, sort_keys=True, default=str)
        signature = hashlib.sha256(canonical_data.encode()).hexdigest()

        return f"SHA256:{signature}"

    async def _transmit_report(self, report: SecurityReport):
        """Transmit report to orchestrator (mock implementation)."""
        # In production, this would use MCP to send to orchestrator
        logger.debug(f"Transmitting report {report.report_id} to orchestrator {self.orchestrator_id}")

        # Simulate network delay
        await asyncio.sleep(0.01)

    async def _cleanup_audit_trail(self, max_reports: int = 1000):
        """Clean up old audit trail entries."""
        if len(self.audit_trail) > max_reports:
            # Keep only the most recent reports
            self.audit_trail = self.audit_trail[-max_reports//2:]
            logger.info("Cleaned up old audit trail entries")

        # Clean up old anomaly alerts (keep last 100)
        if len(self.anomaly_alerts) > 100:
            self.anomaly_alerts = self.anomaly_alerts[-50:]

    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Audit summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_reports = [r for r in self.audit_trail if r.timestamp > cutoff_time]
        recent_anomalies = [a for a in self.anomaly_alerts if a.timestamp > cutoff_time]

        # Count by type and severity
        report_types = {}
        severity_counts = {}

        for report in recent_reports:
            report_type = report.report_type.value
            severity = report.severity.value

            report_types[report_type] = report_types.get(report_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "time_period_hours": hours,
            "total_reports": len(recent_reports),
            "total_anomalies": len(recent_anomalies),
            "reports_by_type": report_types,
            "reports_by_severity": severity_counts,
            "reports_sent": self.reports_sent,
            "reports_failed": self.reports_failed,
            "success_rate": self.reports_sent / max(1, self.reports_sent + self.reports_failed),
            "anomaly_agents": list(set(a.agent_id for a in recent_anomalies)),
            "high_severity_count": severity_counts.get("high", 0) + severity_counts.get("critical", 0)
        }

    def get_agent_anomaly_history(self, agent_id: str) -> List[AnomalyAlert]:
        """Get anomaly history for specific agent."""
        return [a for a in self.anomaly_alerts if a.agent_id == agent_id]

    def get_recent_alerts(self, severity: Optional[Severity] = None, limit: int = 10) -> List[SecurityReport]:
        """Get recent security alerts.

        Args:
            severity: Filter by severity level
            limit: Maximum number of alerts to return

        Returns:
            List of recent security reports
        """
        alerts = [r for r in self.audit_trail if r.report_type == ReportType.SECURITY_ALERT]

        if severity:
            alerts = [r for r in alerts if r.severity == severity]

        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)

        return alerts[:limit]