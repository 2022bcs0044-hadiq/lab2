pipeline {
    agent any

    environment {
        REPO_URL     = "https://github.com/2022bcs0044-hadiq/lab2.git"
        BRANCH_NAME  = "main"
        IMAGE_NAME   = "2022bcs0044hadiqc/wine_predict_2022bcs0044"
    }

    stages {

        stage('Checkout Code') {
            steps {
                git branch: "${BRANCH_NAME}",
                    credentialsId: 'git-creds',
                    url: "${REPO_URL}"
            }
        }

        stage('Setup Python Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Read Model Metrics') {
            steps {
                script {
                    def metrics = readJSON file: 'outputs/evaluation/metrics.json'
                    env.CURRENT_R2 = metrics.RandomForest.R2_Score.toString()
                    echo "Current R2 Score: ${env.CURRENT_R2}"
                }
            }
        }

        stage('Compare With Best Metric') {
            steps {
                script {
                    withCredentials([string(credentialsId: 'best-accuracy', variable: 'BEST_R2')]) {
                        if (env.CURRENT_R2.toFloat() <= BEST_R2.toFloat()) {
                            error("2022BCS0044 ---- Metric did not improve")
                        } else {
                            echo "Metric improved. Proceeding to deployment."
                        }
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'dockerhub-creds') {
                        sh "docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} lab3"
                        sh "docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest"
                    }
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'dockerhub-creds') {
                        sh "docker push ${IMAGE_NAME}:${BUILD_NUMBER}"
                        sh "docker push ${IMAGE_NAME}:latest"
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'outputs/**', fingerprint: true
        }
        success {
            echo "Pipeline completed successfully"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}
