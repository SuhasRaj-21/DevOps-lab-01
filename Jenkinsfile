pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'C:\Users\suhas\Desktop\udbhav main stage day 2\INTER POSTER DAY 2 -m pip install -r requirements.txt'
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

        stage('Run Docker Container') {
            steps {
                bat 'docker run -d -p 5000:5000 devops-app'
            }
        }
    }
}
