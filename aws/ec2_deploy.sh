#!/bin/bash
# =============================================================
# ec2_deploy.sh
# EC2 Instance Setup & Deployment Script
# Customer Churn Prediction — AWS EC2
# =============================================================
# Usage:
#   chmod +x aws/ec2_deploy.sh
#   bash aws/ec2_deploy.sh
# =============================================================

set -e  # Exit on any error

echo "=============================================="
echo "  Customer Churn Prediction — EC2 Deployment"
echo "=============================================="

# ─────────────────────────────────────────────
# CONFIG — Edit these
# ─────────────────────────────────────────────
EC2_USER="ec2-user"
EC2_HOST="your-ec2-public-dns.compute.amazonaws.com"  # ← change this
KEY_PATH="~/.ssh/your-key.pem"                         # ← change this
APP_DIR="/home/ec2-user/churn"
REPO_URL="https://github.com/your-username/customer-churn-prediction.git"

# ─────────────────────────────────────────────
# STEP 1: SSH into EC2 and setup environment
# ─────────────────────────────────────────────
echo ""
echo "Step 1: Setting up EC2 environment..."

ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'

# Update system
sudo yum update -y

# Install Python 3.9
sudo yum install python39 python39-pip git -y

# Create app directory
mkdir -p /home/ec2-user/churn
cd /home/ec2-user/churn

# Clone or pull latest code
if [ -d ".git" ]; then
    echo "Pulling latest code..."
    git pull origin main
else
    echo "Cloning repository..."
    git clone $REPO_URL .
fi

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete"
EOF

echo "✅ Step 1 complete"

# ─────────────────────────────────────────────
# STEP 2: Configure AWS credentials on EC2
# ─────────────────────────────────────────────
echo ""
echo "Step 2: AWS credentials..."
echo "NOTE: Run 'aws configure' manually on EC2 or use IAM Instance Role"
echo "  Recommended: Attach IAM Role with S3 access to EC2 instance"

# ─────────────────────────────────────────────
# STEP 3: Setup cron job for weekly retraining
# ─────────────────────────────────────────────
echo ""
echo "Step 3: Setting up weekly retraining cron job..."

ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'

# Create log directory
sudo mkdir -p /var/log/churn
sudo chown ec2-user:ec2-user /var/log/churn

# Add cron job: runs every Sunday at midnight
CRON_JOB="0 0 * * 0 cd /home/ec2-user/churn && source venv/bin/activate && python src/retrain.py >> /var/log/churn/retrain.log 2>&1"

# Check if cron job already exists
(crontab -l 2>/dev/null | grep -q "retrain.py") && {
    echo "Cron job already exists. Skipping."
} || {
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Cron job added: runs every Sunday at midnight"
}

crontab -l
EOF

echo "✅ Step 3 complete"

# ─────────────────────────────────────────────
# STEP 4: Run initial training
# ─────────────────────────────────────────────
echo ""
echo "Step 4: Running initial model training..."

ssh -i $KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'
cd /home/ec2-user/churn
source venv/bin/activate

echo "Running ETL..."
python src/etl.py --source s3 --destination both

echo "Training models..."
python src/train.py --upload-s3

echo "Evaluating models..."
python src/evaluate.py

echo "✅ Initial training complete!"
EOF

echo ""
echo "=============================================="
echo "  ✅ DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "  EC2 Instance : $EC2_HOST"
echo "  App Directory: $APP_DIR"
echo "  Retrain Cron : Every Sunday midnight"
echo "  Logs         : /var/log/churn/retrain.log"
echo ""
echo "  To check logs:"
echo "  ssh -i $KEY_PATH $EC2_USER@$EC2_HOST 'tail -f /var/log/churn/retrain.log'"
echo ""
