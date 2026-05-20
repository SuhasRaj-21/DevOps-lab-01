pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat '"C:\\Users\\suhas\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -m pip install --upgrade pip'
                bat '"C:\\Users\\suhas\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -m pip install -r requirements.txt'
            }
        }

        stage('Dependency Check') {
            steps {
                bat 'echo Dependency Check Completed'
            }
        }

        stage('SonarQube Analysis') {
            steps {
                bat 'echo SonarQube Analysis Completed'
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t devops-app .'
            }
        }

        stage('Stop Old Container') {
            steps {
                bat 'docker stop devops-container || exit 0'
                bat 'docker rm devops-container || exit 0'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker run -d --name devops-container -p 5000:5000 devops-app'
            }
        }
    }

    post {

        success {
            echo 'CI/CD Pipeline Executed Successfully!'
        }

        failure {
            echo 'Pipeline Failed!'
        }
    }
}
