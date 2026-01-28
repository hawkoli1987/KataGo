#!/bin/bash
# =============================================================================
# KataGo Server Management Script
# =============================================================================
# Manages PBS jobs for candidate and reference KataGo analysis servers.
#
# Usage:
#   ./manage_servers.sh start [candidate|reference|all]
#   ./manage_servers.sh stop [candidate|reference|all]
#   ./manage_servers.sh restart reference --model <model_path>
#   ./manage_servers.sh status
#   ./manage_servers.sh wait [candidate|reference] [timeout_seconds]
#
# Environment:
#   CANDIDATE_MODEL: Override candidate model path
#   REFERENCE_MODEL: Override reference model path
#   CANDIDATE_PORT: Override candidate port (default: 9200)
#   REFERENCE_PORT: Override reference port (default: 9201)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
CONFIG_FILE="${REPO_DIR}/configs/config.yaml"
PBS_SCRIPT="${SCRIPT_DIR}/katago_server.pbs"
PID_DIR="${SCRIPT_DIR}/log/.pids"

# Create PID directory
mkdir -p "${PID_DIR}"

# Read config values using Python
read_config() {
    python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
$1
" 2>/dev/null || echo ""
}

# Default configuration from config.yaml
CANDIDATE_PORT="${CANDIDATE_PORT:-$(read_config "print(cfg['servers']['candidate']['port'])")}"
REFERENCE_PORT="${REFERENCE_PORT:-$(read_config "print(cfg['servers']['reference']['port'])")}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-$(read_config "print(cfg['servers']['candidate']['model_path'])")}"
REFERENCE_MODEL="${REFERENCE_MODEL:-$(read_config "print(cfg['servers']['reference']['model_path'])")}"

# Fallback defaults
CANDIDATE_PORT="${CANDIDATE_PORT:-9200}"
REFERENCE_PORT="${REFERENCE_PORT:-9201}"

# PID files
CANDIDATE_PID_FILE="${PID_DIR}/candidate.jobid"
REFERENCE_PID_FILE="${PID_DIR}/reference.jobid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# Get job ID from PID file
get_job_id() {
    local pid_file="$1"
    if [ -f "${pid_file}" ]; then
        cat "${pid_file}"
    else
        echo ""
    fi
}

# Check if a PBS job is running
is_job_running() {
    local job_id="$1"
    if [ -z "${job_id}" ]; then
        return 1
    fi
    qstat "${job_id}" &>/dev/null
}

# Get hostname where job is running
get_job_host() {
    local job_id="$1"
    local log_dir="${SCRIPT_DIR}/log/${job_id}"
    if [ -f "${log_dir}/pbs.log" ]; then
        grep "^Hostname:" "${log_dir}/pbs.log" 2>/dev/null | head -1 | awk '{print $2}'
    else
        echo ""
    fi
}

# Start a KataGo server
start_server() {
    local server_type="$1"
    local model_path="$2"
    local port="$3"
    local pid_file="$4"
    
    # Check if already running
    local existing_job=$(get_job_id "${pid_file}")
    if is_job_running "${existing_job}"; then
        log "${server_type} server already running (job: ${existing_job})"
        return 0
    fi
    
    log "Starting ${server_type} server on port ${port}..."
    log "Model: ${model_path}"
    
    # Submit PBS job
    local job_id=$(qsub -v "MODEL_PATH=${model_path},SERVER_PORT=${port},SERVER_NAME=${server_type}" "${PBS_SCRIPT}")
    
    if [ -z "${job_id}" ]; then
        error "Failed to submit PBS job"
    fi
    
    # Save job ID
    echo "${job_id}" > "${pid_file}"
    log "${server_type} server submitted: ${job_id}"
    
    echo "${job_id}"
}

# Stop a KataGo server
stop_server() {
    local server_type="$1"
    local pid_file="$2"
    
    local job_id=$(get_job_id "${pid_file}")
    
    if [ -z "${job_id}" ]; then
        log "${server_type} server not running (no job ID found)"
        return 0
    fi
    
    if is_job_running "${job_id}"; then
        log "Stopping ${server_type} server (job: ${job_id})..."
        qdel "${job_id}" 2>/dev/null || true
        rm -f "${pid_file}"
        log "${server_type} server stopped"
    else
        log "${server_type} server not running"
        rm -f "${pid_file}"
    fi
}

# Wait for server to be healthy
wait_for_server() {
    local host="$1"
    local port="$2"
    local timeout="${3:-120}"
    local start_time=$(date +%s)
    
    log "Waiting for server at ${host}:${port} (timeout: ${timeout}s)..."
    
    while true; do
        local elapsed=$(($(date +%s) - start_time))
        if [ ${elapsed} -ge ${timeout} ]; then
            error "Timeout waiting for server at ${host}:${port}"
        fi
        
        if curl -s "http://${host}:${port}/health" &>/dev/null; then
            log "Server at ${host}:${port} is healthy"
            return 0
        fi
        
        sleep 2
    done
}

# Wait for a server type to be ready
wait_for_server_type() {
    local server_type="$1"
    local timeout="${2:-120}"
    
    local pid_file port
    if [ "${server_type}" = "candidate" ]; then
        pid_file="${CANDIDATE_PID_FILE}"
        port="${CANDIDATE_PORT}"
    else
        pid_file="${REFERENCE_PID_FILE}"
        port="${REFERENCE_PORT}"
    fi
    
    local job_id=$(get_job_id "${pid_file}")
    if [ -z "${job_id}" ]; then
        error "${server_type} server not started"
    fi
    
    # Wait for job to start running and get hostname
    local start_time=$(date +%s)
    local host=""
    while [ -z "${host}" ]; do
        local elapsed=$(($(date +%s) - start_time))
        if [ ${elapsed} -ge ${timeout} ]; then
            error "Timeout waiting for ${server_type} job to start"
        fi
        
        if ! is_job_running "${job_id}"; then
            error "${server_type} job ${job_id} is not running"
        fi
        
        host=$(get_job_host "${job_id}")
        if [ -z "${host}" ]; then
            sleep 2
        fi
    done
    
    wait_for_server "${host}" "${port}" "$((timeout - ($(date +%s) - start_time)))"
    
    # Output endpoint info
    echo "http://${host}:${port}"
}

# Show status of servers
show_status() {
    echo "=== KataGo Server Status ==="
    echo ""
    
    for server_type in candidate reference; do
        local pid_file port
        if [ "${server_type}" = "candidate" ]; then
            pid_file="${CANDIDATE_PID_FILE}"
            port="${CANDIDATE_PORT}"
        else
            pid_file="${REFERENCE_PID_FILE}"
            port="${REFERENCE_PORT}"
        fi
        
        local job_id=$(get_job_id "${pid_file}")
        
        if [ -z "${job_id}" ]; then
            echo "${server_type}: NOT STARTED"
        elif is_job_running "${job_id}"; then
            local host=$(get_job_host "${job_id}")
            if [ -n "${host}" ]; then
                local health_status="UNKNOWN"
                if curl -s "http://${host}:${port}/health" &>/dev/null; then
                    health_status="HEALTHY"
                else
                    health_status="NOT READY"
                fi
                echo "${server_type}: RUNNING (job: ${job_id}, endpoint: http://${host}:${port}, status: ${health_status})"
            else
                echo "${server_type}: STARTING (job: ${job_id})"
            fi
        else
            echo "${server_type}: STOPPED (stale job: ${job_id})"
            rm -f "${pid_file}"
        fi
    done
}

# Print endpoint URLs
print_endpoints() {
    for server_type in candidate reference; do
        local pid_file port
        if [ "${server_type}" = "candidate" ]; then
            pid_file="${CANDIDATE_PID_FILE}"
            port="${CANDIDATE_PORT}"
        else
            pid_file="${REFERENCE_PID_FILE}"
            port="${REFERENCE_PORT}"
        fi
        
        local job_id=$(get_job_id "${pid_file}")
        if [ -n "${job_id}" ] && is_job_running "${job_id}"; then
            local host=$(get_job_host "${job_id}")
            if [ -n "${host}" ]; then
                echo "${server_type^^}_ENDPOINT=http://${host}:${port}"
            fi
        fi
    done
}

# Main command handling
case "${1:-help}" in
    start)
        target="${2:-all}"
        case "${target}" in
            candidate)
                start_server "candidate" "${CANDIDATE_MODEL}" "${CANDIDATE_PORT}" "${CANDIDATE_PID_FILE}"
                ;;
            reference)
                start_server "reference" "${REFERENCE_MODEL}" "${REFERENCE_PORT}" "${REFERENCE_PID_FILE}"
                ;;
            all)
                start_server "candidate" "${CANDIDATE_MODEL}" "${CANDIDATE_PORT}" "${CANDIDATE_PID_FILE}"
                start_server "reference" "${REFERENCE_MODEL}" "${REFERENCE_PORT}" "${REFERENCE_PID_FILE}"
                ;;
            *)
                error "Unknown target: ${target}. Use: candidate, reference, or all"
                ;;
        esac
        ;;
    
    stop)
        target="${2:-all}"
        case "${target}" in
            candidate)
                stop_server "candidate" "${CANDIDATE_PID_FILE}"
                ;;
            reference)
                stop_server "reference" "${REFERENCE_PID_FILE}"
                ;;
            all)
                stop_server "candidate" "${CANDIDATE_PID_FILE}"
                stop_server "reference" "${REFERENCE_PID_FILE}"
                ;;
            *)
                error "Unknown target: ${target}. Use: candidate, reference, or all"
                ;;
        esac
        ;;
    
    restart)
        target="${2:-}"
        if [ -z "${target}" ]; then
            error "Usage: $0 restart [candidate|reference] [--model <path>]"
        fi
        
        # Parse --model argument
        shift 2
        model_path=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --model)
                    model_path="$2"
                    shift 2
                    ;;
                *)
                    error "Unknown option: $1"
                    ;;
            esac
        done
        
        case "${target}" in
            candidate)
                stop_server "candidate" "${CANDIDATE_PID_FILE}"
                sleep 2
                model="${model_path:-${CANDIDATE_MODEL}}"
                start_server "candidate" "${model}" "${CANDIDATE_PORT}" "${CANDIDATE_PID_FILE}"
                ;;
            reference)
                stop_server "reference" "${REFERENCE_PID_FILE}"
                sleep 2
                model="${model_path:-${REFERENCE_MODEL}}"
                start_server "reference" "${model}" "${REFERENCE_PORT}" "${REFERENCE_PID_FILE}"
                ;;
            *)
                error "Unknown target: ${target}. Use: candidate or reference"
                ;;
        esac
        ;;
    
    wait)
        target="${2:-all}"
        timeout="${3:-120}"
        case "${target}" in
            candidate)
                wait_for_server_type "candidate" "${timeout}"
                ;;
            reference)
                wait_for_server_type "reference" "${timeout}"
                ;;
            all)
                wait_for_server_type "candidate" "${timeout}"
                wait_for_server_type "reference" "${timeout}"
                ;;
            *)
                error "Unknown target: ${target}. Use: candidate, reference, or all"
                ;;
        esac
        ;;
    
    status)
        show_status
        ;;
    
    endpoints)
        print_endpoints
        ;;
    
    help|--help|-h)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start [candidate|reference|all]   Start server(s)"
        echo "  stop [candidate|reference|all]    Stop server(s)"
        echo "  restart <target> [--model <path>] Restart server with optional new model"
        echo "  wait [candidate|reference|all]    Wait for server(s) to be healthy"
        echo "  status                            Show server status"
        echo "  endpoints                         Print endpoint URLs (for sourcing)"
        echo ""
        echo "Environment variables:"
        echo "  CANDIDATE_MODEL  Model path for candidate server"
        echo "  REFERENCE_MODEL  Model path for reference server"
        echo "  CANDIDATE_PORT   Port for candidate server (default: 9200)"
        echo "  REFERENCE_PORT   Port for reference server (default: 9201)"
        ;;
    
    *)
        error "Unknown command: $1. Use '$0 help' for usage."
        ;;
esac
