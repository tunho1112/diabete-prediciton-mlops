#!/usr/bin/env python3
"""
Discord Webhook Proxy for Alertmanager
Converts Prometheus alert format to Discord webhook format
"""
import json
import sys
import requests
from typing import Dict, Any

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1449404485142056994/TRgvGYD1tM1fLXOxj21N4Rsc5UsZGYAcImteMbj6Qdn2L2TZt58p23tvSP1o9HSCt-1O"

def format_discord_message(alert_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Prometheus alert format to Discord format"""
    alerts = alert_data.get("alerts", [])
    
    if not alerts:
        return {"content": "No alerts"}
    
    # Build message content
    content_parts = []
    embeds = []
    
    for alert in alerts:
        status = alert.get("status", "unknown").upper()
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        
        alertname = labels.get("alertname", "Unknown Alert")
        severity = labels.get("severity", "unknown")
        summary = annotations.get("summary", "")
        description = annotations.get("description", "")
        
        # Create embed for each alert
        color = 0xFF0000 if status == "FIRING" else 0x00FF00  # Red for firing, green for resolved
        
        embed = {
            "title": f"{status}: {alertname}",
            "description": f"**Summary:** {summary}\n\n**Description:** {description}",
            "color": color,
            "fields": [
                {"name": "Severity", "value": severity, "inline": True},
                {"name": "Status", "value": status, "inline": True}
            ],
            "timestamp": alert.get("startsAt", "")
        }
        embeds.append(embed)
    
    return {
        "content": f"🚨 **Alertmanager Notification** ({len(alerts)} alert{'s' if len(alerts) > 1 else ''})",
        "embeds": embeds
    }

def main():
    """Main function to process webhook request"""
    try:
        # Read alert data from stdin (Alertmanager sends JSON)
        alert_data = json.load(sys.stdin)
        
        # Format for Discord
        discord_payload = format_discord_message(alert_data)
        
        # Send to Discord
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=discord_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 204:
            print("Successfully sent alert to Discord", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"Failed to send to Discord: {response.status_code} - {response.text}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error processing alert: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

